"""
Step 4: Face detection + tracking.

Detects faces (MTCNN) at a fixed sample rate and links detections into
tracks using greedy IOU matching. Tracks are reset at every shot boundary
(from step 1) because a track spanning a cut is not a valid assumption --
a cut can swap who's on screen entirely.

NOTE ON DETECTOR CHOICE: this uses MTCNN (pip-installable, weights ship
inside the wheel) because this sandbox's network is restricted and cannot
reach Google Drive or git-lfs-hosted model files, which is where most
stronger detectors (YuNet, RetinaFace) host their weights. MTCNN was
validated against this footage: 4/4 spot-checked frames with angled/profile
faces detected at >0.95 confidence, vs. 1/4 for a tuned Haar Cascade
baseline. On a machine with full internet access, swap `detect_faces()`
for a GPU-accelerated detector (RetinaFace, YuNet, insightface) -- the
tracking/output logic downstream doesn't need to change, since it only
depends on getting (box, keypoints) per frame.

Usage:
    python3 04_track_faces.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/shots/<video_id>_shots.json
Writes: data/episodes/<video_id>_face_tracks.json
"""
import argparse
import json
import os

import cv2
import numpy as np
from mtcnn import MTCNN

SAMPLE_FPS = 5.0
IOU_MATCH_THRESH = 0.3   # min IOU to link a detection to an existing track
MAX_MISSED_FRAMES = 2    # allow a track to survive this many sampled frames with no match
MIN_TRACK_DETECTIONS = 3  # drop tracks shorter than this (likely spurious)


def box_iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_faces_in_shot(cap, native_fps, shot, detector):
    """Runs MTCNN at SAMPLE_FPS across one shot's time range."""
    start_t, end_t = shot["start_sec"], shot["end_sec"]
    step_t = 1.0 / SAMPLE_FPS
    detections = []  # list of (t, [ {box, keypoints, confidence}, ... ])
    t = start_t
    while t < end_t:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * native_fps)))
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        detections.append((round(t, 3), faces))
        t += step_t
    return detections


def link_tracks(detections):
    """Greedy IOU tracker over a single shot's per-frame detections."""
    active_tracks = []  # each: {"track_id", "dets": [...], "missed": n}
    finished_tracks = []
    next_id = 0

    for t, faces in detections:
        unmatched_faces = list(range(len(faces)))
        for track in active_tracks:
            if not track["dets"]:
                continue
            last_box = track["dets"][-1]["box"]
            best_iou, best_j = 0.0, None
            for j in unmatched_faces:
                iou = box_iou(last_box, faces[j]["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= IOU_MATCH_THRESH:
                f = faces[best_j]
                track["dets"].append({
                    "t": t, "box": f["box"], "keypoints": f["keypoints"],
                    "confidence": float(f["confidence"]),
                })
                track["missed"] = 0
                unmatched_faces.remove(best_j)
            else:
                track["missed"] += 1

        for j in unmatched_faces:
            f = faces[j]
            active_tracks.append({
                "track_id": next_id,
                "dets": [{"t": t, "box": f["box"], "keypoints": f["keypoints"],
                          "confidence": float(f["confidence"])}],
                "missed": 0,
            })
            next_id += 1

        still_active = []
        for track in active_tracks:
            if track["missed"] > MAX_MISSED_FRAMES:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

    finished_tracks.extend(active_tracks)
    return [tr for tr in finished_tracks if len(tr["dets"]) >= MIN_TRACK_DETECTIONS]


def serialize_box(box):
    return [int(v) for v in box]


def serialize_keypoints(kp):
    return {k: [int(v[0]), int(v[1])] for k, v in kp.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    shots_path = os.path.join(args.data_dir, "shots", f"{args.video_id}_shots.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_tracks.json")

    with open(shots_path, encoding="utf-8") as f:
        shots = json.load(f)

    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    all_shot_tracks = []
    for shot in shots:
        detections = detect_faces_in_shot(cap, native_fps, shot, detector)
        tracks = link_tracks(detections)
        shot_out = {
            "shot_id": shot["shot_id"],
            "start_sec": shot["start_sec"],
            "end_sec": shot["end_sec"],
            "tracks": [],
        }
        for tr in tracks:
            shot_out["tracks"].append({
                "track_id": tr["track_id"],
                "n_detections": len(tr["dets"]),
                "detections": [{
                    "t": d["t"],
                    "box": serialize_box(d["box"]),
                    "keypoints": serialize_keypoints(d["keypoints"]),
                    "confidence": round(d["confidence"], 4),
                } for d in tr["dets"]],
            })
        all_shot_tracks.append(shot_out)
        print(f"shot {shot['shot_id']:02d} [{shot['start_sec']:.2f}-{shot['end_sec']:.2f}s]: "
              f"{len(tracks)} track(s), "
              + ", ".join(f"id{tr['track_id']}(n={len(tr['dets'])})" for tr in tracks))
    cap.release()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "sample_fps": SAMPLE_FPS, "shots": all_shot_tracks},
                   f, indent=2, ensure_ascii=False)
    print(f"\nFace tracks -> {out_path}")


if __name__ == "__main__":
    main()
