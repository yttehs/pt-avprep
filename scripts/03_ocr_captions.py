"""
Step 3: OCR the caption segments found in step 2.

For each caption segment, grabs a representative frame (the middle of the
stable window -- avoids transition frames at the on/off edges), crops to
the Portuguese line, lightly preprocesses, and runs Tesseract (por).

Also OCRs the English line for reference/QA (not used for diarization).

Usage:
    python3 03_ocr_captions.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/episodes/<video_id>_caption_segments.json
Writes: data/episodes/<video_id>_transcript.json
        outputs/<video_id>_ocr_review.png  (contact sheet for spot-checking)
"""
import argparse
import json
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image

PT_LINE_FRAC = (0.0, 282 / 360, 1.0, 307 / 360)
EN_LINE_FRAC = (0.0, 308 / 360, 1.0, 326 / 360)

WHITE_THRESH = 190  # same value validated in 02_segment_captions.py
UPSCALE = 3  # tesseract does much better on small text when upscaled


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


def ocr_line(frame, frac_box, lang):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = frac_box_to_px(frac_box, w, h)
    crop = frame[y0:y1, x0:x1]
    proc = preprocess_for_ocr(crop)
    # PSM 7 = treat the image as a single text line.
    config = "--psm 7"
    text = pytesseract.image_to_string(proc, lang=lang, config=config).strip()
    return text, crop, proc


def get_frame_at(cap, t_sec, native_fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_sec * native_fps)))
    ok, frame = cap.read()
    return frame if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--outputs-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = parser.parse_args()

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
    for seg in segments:
        mid_t = (seg["start_sec"] + seg["end_sec"]) / 2.0
        frame = get_frame_at(cap, mid_t, native_fps)
        if frame is None:
            continue
        pt_text, pt_crop, pt_proc = ocr_line(frame, PT_LINE_FRAC, "por")
        en_text, _, _ = ocr_line(frame, EN_LINE_FRAC, "eng")

        results.append({
            "segment_id": seg["segment_id"],
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "mid_sec": round(mid_t, 2),
            "text_pt": pt_text,
            "text_en_ref": en_text,
        })
        review_crops.append((seg["segment_id"], seg["start_sec"], seg["end_sec"], pt_crop, pt_text))
    cap.release()

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
