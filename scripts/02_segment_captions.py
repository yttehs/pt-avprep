"""
Step 2: Caption segmentation.

Finds stable caption "cards" (onset/offset timestamps) by tracking a
white-text-pixel mask in the caption crop region, without OCR-ing every
frame. Two signals drive segmentation:
  1. text_density: fraction of near-white pixels in the crop -> is a
     caption present at all right now?
  2. mask_similarity vs. the segment's reference mask -> did the caption
     card change (new text) even though text is continuously present?

Crop coordinates are stored as FRACTIONS of frame width/height so they
generalize across resolutions, calibrated here against a 640x360 frame:
  PT line:  x in [0.0, 1.0], y in [282/360, 307/360]
  EN line:  x in [0.0, 1.0], y in [308/360, 326/360]

Usage:
    python3 02_segment_captions.py <video_id>

Reads:  data/raw/<video_id>.mp4
Writes: data/episodes/<video_id>_caption_segments.json
        outputs/<video_id>_caption_signal.png  (debug plot)
"""
import argparse
import json
import os

import cv2
import numpy as np

# Calibrated on 640x360 Easy Portuguese street-interview template.
PT_LINE_FRAC = (0.0, 282 / 360, 1.0, 307 / 360)  # x0, y0, x1, y1
EN_LINE_FRAC = (0.0, 308 / 360, 1.0, 326 / 360)

WHITE_THRESH = 190          # grayscale value above which a pixel counts as "text"
DENSITY_ON_THRESH = 0.012   # fraction of white pixels in crop; cheap first-pass candidate gate
SIM_THRESH = 0.55           # IoU below this vs. reference mask => new caption card
MIN_SEGMENT_SEC = 0.3       # drop segments shorter than this (likely flicker/noise)

# Shape-based text gate. Raw white-pixel density alone is fooled by bright scene
# content (sunlit pavement, water spray, glare) that happens to fall in the crop
# band. Real caption text forms many small letter-shaped connected components at
# a consistent height (font x-height); scene glare forms few, larger/thinner blobs.
# Calibrated on this template: real captions -> 20-58 components, height ~10px;
# false positives (pavement glare, fountain spray) -> <=9 text-like components,
# median height ~2px.
TEXT_LIKE_MIN_H = 4
TEXT_LIKE_MAX_H = 22
TEXT_LIKE_MAX_AREA = 250
MIN_TEXT_LIKE_COMPONENTS = 10


def frac_box_to_px(frac_box, w, h):
    x0, y0, x1, y1 = frac_box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def white_mask(gray_crop):
    return (gray_crop > WHITE_THRESH).astype(np.uint8)


def count_text_like_components(mask):
    """Returns (n_text_like_components, total_components). mask is 0/1 uint8."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return 0, 0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    text_like = np.sum(
        (heights >= TEXT_LIKE_MIN_H) & (heights <= TEXT_LIKE_MAX_H) & (areas <= TEXT_LIKE_MAX_AREA)
    )
    return int(text_like), n_labels - 1


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty -> treat as identical (shouldn't happen when density gate is on)
    return inter / union


def segment_captions(video_path, sample_fps=8.0):
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / native_fps if native_fps else None

    pt_box = frac_box_to_px(PT_LINE_FRAC, w, h)
    step = max(1, round(native_fps / sample_fps))

    signal = []  # (t_sec, density, mask, n_text_like)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            t = frame_idx / native_fps
            x0, y0, x1, y1 = pt_box
            crop = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = white_mask(gray)
            density = mask.mean()
            n_text_like = 0
            if density >= DENSITY_ON_THRESH:
                n_text_like, _ = count_text_like_components(mask)
            signal.append((t, density, mask, n_text_like))
        frame_idx += 1
    cap.release()

    # Walk the signal, grouping into segments.
    segments = []
    cur = None  # dict with start_t, end_t, ref_mask, densities
    for t, density, mask, n_text_like in signal:
        present = density >= DENSITY_ON_THRESH and n_text_like >= MIN_TEXT_LIKE_COMPONENTS
        if not present:
            if cur is not None:
                segments.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {"start_t": t, "end_t": t, "ref_mask": mask, "densities": [density]}
            continue
        sim = mask_iou(cur["ref_mask"], mask)
        if sim < SIM_THRESH:
            # caption card changed -> close current, open new
            segments.append(cur)
            cur = {"start_t": t, "end_t": t, "ref_mask": mask, "densities": [density]}
        else:
            cur["end_t"] = t
            cur["densities"].append(density)
    if cur is not None:
        segments.append(cur)

    # Filter tiny/noisy segments and drop the mask arrays before serializing.
    out = []
    frame_dt = step / native_fps
    for i, seg in enumerate(segments):
        dur = seg["end_t"] - seg["start_t"] + frame_dt  # add one frame width so single-sample segments aren't 0-length
        if dur < MIN_SEGMENT_SEC:
            continue
        out.append({
            "segment_id": len(out),
            "start_sec": round(seg["start_t"], 2),
            "end_sec": round(seg["end_t"] + frame_dt, 2),
            "mean_density": round(float(np.mean(seg["densities"])), 4),
            "n_samples": len(seg["densities"]),
        })

    meta = {
        "video_path": video_path,
        "width": w, "height": h,
        "native_fps": native_fps,
        "sample_fps_requested": sample_fps,
        "sample_fps_actual": native_fps / step,
        "duration_sec": duration,
        "pt_line_frac": PT_LINE_FRAC,
        "en_line_frac": EN_LINE_FRAC,
        "params": {
            "white_thresh": WHITE_THRESH,
            "density_on_thresh": DENSITY_ON_THRESH,
            "sim_thresh": SIM_THRESH,
            "min_segment_sec": MIN_SEGMENT_SEC,
        },
    }
    return out, meta, signal


def save_debug_plot(signal, segments, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [s[0] for s in signal]
    ds = [s[1] for s in signal]
    tl = [s[3] for s in signal]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    ax.plot(ts, ds, linewidth=0.8, color="steelblue", label="text density")
    ax.axhline(DENSITY_ON_THRESH, color="gray", linestyle="--", linewidth=0.7)
    for seg in segments:
        ax.axvspan(seg["start_sec"], seg["end_sec"], color="orange", alpha=0.25)
    ax.set_ylabel("white pixel density")
    ax.set_title("Caption text-presence signal with detected segments (orange)")
    ax.legend(loc="upper right")

    ax2.plot(ts, tl, linewidth=0.8, color="darkgreen", label="text-like components")
    ax2.axhline(MIN_TEXT_LIKE_COMPONENTS, color="gray", linestyle="--", linewidth=0.7)
    for seg in segments:
        ax2.axvspan(seg["start_sec"], seg["end_sec"], color="orange", alpha=0.25)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("# text-like components")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Debug plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--outputs-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_caption_segments.json")
    plot_path = os.path.join(args.outputs_dir, f"{args.video_id}_caption_signal.png")

    segments, meta, signal = segment_captions(video_path, sample_fps=args.sample_fps)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "segments": segments}, f, indent=2, ensure_ascii=False)

    print(f"Detected {len(segments)} caption segments -> {out_path}")
    for s in segments:
        dur = round(s["end_sec"] - s["start_sec"], 2)
        print(f"  seg {s['segment_id']:03d}: {s['start_sec']:7.2f}s - {s['end_sec']:7.2f}s "
              f"(dur {dur}s, density {s['mean_density']}, n={s['n_samples']})")

    os.makedirs(args.outputs_dir, exist_ok=True)
    save_debug_plot(signal, segments, plot_path)


if __name__ == "__main__":
    main()
