"""
Step 3: OCR the caption segments found in step 2.

For each caption segment, grabs a representative frame (the middle of the
stable window -- avoids transition frames at the on/off edges), crops to
the Portuguese line, lightly preprocesses, and runs Tesseract (por).

Also OCRs the English line for reference/QA (not used for diarization).

Two cleanup passes run after OCR, both added after being observed on real
full-length-video output (not hypothetical):

  1. Low-confidence rejection. Step 2's shape-based caption gate occasionally
     fires on non-caption graphics (e.g. an outro/subscribe screen with
     logos or icons) that happen to look letter-shaped enough to pass the
     connected-component check. Real rendered captions get very high
     Tesseract word confidence (~96 in spot checks); forced OCR on
     non-text graphics scores far lower (~19 in a synthetic test, or empty
     output on pure noise) -- a well-separated, principled signal, checked
     empirically rather than assumed. Segments below MIN_OCR_CONFIDENCE, or
     with empty text, are dropped.
  2. Adjacent near-duplicate merging. A caption's fade-in animation can get
     caught mid-transition as its own short, garbled segment immediately
     before the fully-rendered version (e.g. observed: "episortio" at
     338.07-338.47s immediately followed by the correct "episódio" at
     338.47-340.34s). Adjacent segments with high text similarity get
     merged, keeping the higher-confidence segment's text.

Usage:
    python3 03_ocr_captions.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/episodes/<video_id>_caption_segments.json
Writes: data/episodes/<video_id>_transcript.json
        outputs/<video_id>_ocr_review.png  (contact sheet for spot-checking)
"""
import argparse
import difflib
import json
import os

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

PT_LINE_FRAC = (0.0, 282 / 360, 1.0, 307 / 360)
EN_LINE_FRAC = (0.0, 308 / 360, 1.0, 326 / 360)

WHITE_THRESH = 190  # same value validated in 02_segment_captions.py
UPSCALE = 3  # tesseract does much better on small text when upscaled

# Calibrated empirically (see docstring): real captions ~96, non-text graphics
# forced through OCR ~19 or empty. 60 sits comfortably in the gap between them.
MIN_OCR_CONFIDENCE = 60

# For the adjacent-duplicate merge pass.
MERGE_MAX_GAP_SEC = 0.5     # segments touching or nearly touching in time
MERGE_MIN_SIMILARITY = 0.7  # difflib ratio on the OCR'd text


def frac_box_to_px(frac_box, w, h):
    x0, y0, x1, y1 = frac_box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def preprocess_for_ocr(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    # Fixed threshold, not Otsu. Otsu adapts to each frame's own histogram,
    # which fails when the background behind the semi-transparent caption box
    # has bright elements (archways, railings, sky) comparable in brightness
    # to the text itself -- Otsu then pulls those in as "foreground" and
    # corrupts the mask. We already validated (step 2) that caption text is
    # reliably >190 in grayscale and the box background is reliably well
    # below that, regardless of what's behind the box, so a fixed threshold
    # is the more robust choice here.
    _, binary = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)  # tesseract prefers dark text on light background
    return binary


def ocr_line(frame, frac_box, lang, tessdata_dir=None):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = frac_box_to_px(frac_box, w, h)
    crop = frame[y0:y1, x0:x1]
    proc = preprocess_for_ocr(crop)
    # PSM 7 = treat the image as a single text line.
    config = "--psm 7"
    if tessdata_dir:
        # Lets tesseract find language data in a user-writable directory instead
        # of a system tessdata path -- useful when tesseract itself was installed
        # without root (conda, a local prefix build, etc.) and language packs
        # were dropped in manually rather than via a system package.
        config += f' --tessdata-dir "{tessdata_dir}"'
    data = pytesseract.image_to_data(proc, lang=lang, config=config, output_type=Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confs = [int(c) for c in data["conf"] if c not in ("-1", -1) and int(c) >= 0]
    text = " ".join(words).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf, crop, proc


def merge_near_duplicate_segments(results):
    """Merges adjacent segments whose OCR'd text is a near-duplicate of each
    other -- the fade-in-transition artifact described in the module
    docstring. Keeps the higher-confidence segment's text and boundaries,
    extended to cover the merged segment's full time range."""
    if not results:
        return results
    merged = [dict(results[0])]
    for cur in results[1:]:
        prev = merged[-1]
        gap = cur["start_sec"] - prev["end_sec"]
        similarity = difflib.SequenceMatcher(None, prev["text_pt"], cur["text_pt"]).ratio()
        if gap <= MERGE_MAX_GAP_SEC and similarity >= MERGE_MIN_SIMILARITY:
            keep_cur = cur.get("confidence", 0) >= prev.get("confidence", 0)
            winner = cur if keep_cur else prev
            merged[-1] = {
                **winner,
                "start_sec": prev["start_sec"],
                "end_sec": cur["end_sec"],
                "mid_sec": round((prev["start_sec"] + cur["end_sec"]) / 2, 2),
            }
            print(f"  merged near-duplicate: [{prev['start_sec']:.2f}-{prev['end_sec']:.2f}s] "
                  f"\"{prev['text_pt']}\" + [{cur['start_sec']:.2f}-{cur['end_sec']:.2f}s] "
                  f"\"{cur['text_pt']}\" -> kept \"{winner['text_pt']}\"")
        else:
            merged.append(dict(cur))
    return merged


def get_frame_at(cap, t_sec, native_fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_sec * native_fps)))
    ok, frame = cap.read()
    return frame if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--outputs-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    parser.add_argument("--tesseract-cmd", default=None,
                         help="Path to the tesseract binary, if it's not on PATH "
                              "(e.g. a conda env or a local no-root install).")
    parser.add_argument("--tessdata-dir", default=None,
                         help="Directory containing .traineddata language files, if "
                              "installed outside tesseract's default system location "
                              "(e.g. downloaded manually into a user-writable folder).")
    args = parser.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    seg_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_caption_segments.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_transcript.json")
    review_path = os.path.join(args.outputs_dir, f"{args.video_id}_ocr_review.png")

    with open(seg_path, encoding="utf-8") as f:
        seg_data = json.load(f)
    segments = seg_data["segments"]

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    results = []
    review_crops = []
    n_dropped = 0
    for seg in segments:
        mid_t = (seg["start_sec"] + seg["end_sec"]) / 2.0
        frame = get_frame_at(cap, mid_t, native_fps)
        if frame is None:
            continue
        pt_text, pt_conf, pt_crop, pt_proc = ocr_line(frame, PT_LINE_FRAC, "por", args.tessdata_dir)
        en_text, _, _, _ = ocr_line(frame, EN_LINE_FRAC, "eng", args.tessdata_dir)

        if not pt_text or pt_conf < MIN_OCR_CONFIDENCE:
            n_dropped += 1
            print(f"  dropped low-confidence segment [{seg['start_sec']:.2f}-{seg['end_sec']:.2f}s] "
                  f"conf={pt_conf:.0f} text={pt_text!r}")
            continue

        results.append({
            "segment_id": seg["segment_id"],
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "mid_sec": round(mid_t, 2),
            "text_pt": pt_text,
            "text_en_ref": en_text,
            "confidence": round(pt_conf, 1),
        })
        review_crops.append((seg["segment_id"], seg["start_sec"], seg["end_sec"], pt_crop, pt_text))
    cap.release()

    print(f"\nDropped {n_dropped} low-confidence/empty segment(s) (threshold={MIN_OCR_CONFIDENCE})")

    results = merge_near_duplicate_segments(results)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "transcript": results}, f, indent=2, ensure_ascii=False)

    print(f"OCR'd {len(results)} segments -> {out_path}\n")
    for r in results:
        print(f"  [{r['start_sec']:6.2f}-{r['end_sec']:6.2f}s] {r['text_pt']}")

    # Build a contact sheet: crop + OCR text, for fast human spot-checking.
    os.makedirs(args.outputs_dir, exist_ok=True)
    build_review_sheet(review_crops, review_path)
    print(f"\nReview sheet -> {review_path}")


def build_review_sheet(review_crops, out_path):
    from PIL import ImageDraw

    pad = 6
    row_h = 34
    max_w = max(c[3].shape[1] for c in review_crops)
    total_h = len(review_crops) * (row_h + pad)
    sheet = Image.new("RGB", (max_w + 20, total_h + 20), "white")
    draw = ImageDraw.Draw(sheet)

    y = 10
    for seg_id, start, end, crop_bgr, text in review_crops:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_img = Image.fromarray(crop_rgb)
        sheet.paste(crop_img, (10, y))
        draw.text((10, y + crop_img.height + 1),
                   f"[{seg_id:03d}] {start:.2f}-{end:.2f}s -> {text}",
                   fill="black")
        y += row_h + pad
    sheet.save(out_path)


if __name__ == "__main__":
    main()
