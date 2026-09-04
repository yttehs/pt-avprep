"""
Step 1: Shot boundary detection.

Splits a video into shots (continuous camera takes) using content-aware
cut detection. This matters for later stages because face tracking and
active-speaker detection assume a continuous shot -- both should reset
at every cut.

Usage:
    python3 01_detect_shots.py <video_id>

Reads:  data/raw/<video_id>.mp4
Writes: data/shots/<video_id>_shots.json
"""
import argparse
import json
import os

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_shots(video_path, threshold=27.0, min_scene_len=8):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_id": i,
            "start_sec": round(start.get_seconds(), 3),
            "end_sec": round(end.get_seconds(), 3),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
        })
    return shots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", help="e.g. portuguese_01_clip90 (no extension)")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--threshold", type=float, default=27.0,
                         help="Higher = less sensitive to cuts. PySceneDetect default is 27.")
    parser.add_argument("--min-scene-len", type=int, default=8,
                         help="Minimum shot length in frames.")
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    out_path = os.path.join(args.data_dir, "shots", f"{args.video_id}_shots.json")

    shots = detect_shots(video_path, threshold=args.threshold, min_scene_len=args.min_scene_len)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    print(f"Detected {len(shots)} shots -> {out_path}")
    for s in shots:
        dur = round(s["end_sec"] - s["start_sec"], 2)
        print(f"  shot {s['shot_id']:02d}: {s['start_sec']:7.2f}s - {s['end_sec']:7.2f}s  (dur {dur}s)")


if __name__ == "__main__":
    main()
