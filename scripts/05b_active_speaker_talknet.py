"""
Step 5 (real model): active speaker attribution using pretrained TalkNet-ASD.

Requires a CUDA GPU (TalkNet-ASD's model classes call .cuda() unconditionally)
and full internet access (to `pip install gdown` and pull the ~50MB pretrained
checkpoint from Google Drive) -- meant to run on your server, not the sandbox
this pipeline was prototyped in.

WHERE THIS CODE CAME FROM: the crop/preprocessing math below (crop_video,
extract_features, the duration-ensemble scoring loop) is ported as faithfully
as possible from the official repo's own demoTalkNet.py
(https://github.com/TaoRuijie/TalkNet-ASD), NOT reimplemented from memory --
that file was cloned and read directly to get the exact crop scale, median-
filter smoothing, MFCC window/step, and audio/video alignment constants
right, since small transcription errors in preprocessing would silently
degrade a pretrained model's output with no obvious error. Only these two
things are genuinely different from the original demo:

  1. Face source: the original demo runs its own S3FD face detector + its
     own PySceneDetect scene split internally. This script instead reuses
     the face tracks and shots already produced by this pipeline's own
     04_track_faces.py (MTCNN) and 01_detect_shots.py -- already validated
     against this footage, no reason to duplicate/replace them.
  2. Output: the original demo visualizes scores as an overlay video. This
     script instead averages each track's per-frame scores within each
     caption segment's time window and assigns the segment to the
     highest-scoring track -- i.e. it plugs the real model's output into
     the same "candidates -> assign" structure 05_active_speaker.py used
     for the heuristic, so the two are directly comparable.

SETUP (on your GPU server):
    git clone https://github.com/TaoRuijie/TalkNet-ASD third_party/TalkNet-ASD
    pip install torch torchvision torchaudio  # match your CUDA version
    pip install gdown python_speech_features pandas scipy scikit-learn
    # weights are auto-downloaded on first run (~50MB, Google Drive)

Usage:
    python3 05b_active_speaker_talknet.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/shots/<video_id>_shots.json
        data/episodes/<video_id>_face_tracks.json
        data/episodes/<video_id>_transcript.json
Writes: data/episodes/<video_id>_speaker_attribution_talknet.json
        data/talknet_work/<video_id>/...  (intermediate crops, kept for inspection/debugging)
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

TALKNET_REPO = os.path.join(os.path.dirname(__file__), "..", "third_party", "TalkNet-ASD")
PRETRAIN_GDRIVE_ID = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"  # from the repo's own demoTalkNet.py
PRETRAIN_FILENAME = "pretrain_TalkSet.model"

TARGET_FPS = 25          # TalkNet-ASD's whole pipeline assumes 25fps video / 100fps MFCC (4:1 ratio)
CROP_SCALE = 0.40        # same default as demoTalkNet.py's --cropScale
MEDFILT_KERNEL = 13      # same as demoTalkNet.py's crop_video smoothing
DURATION_SET = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]  # same ensemble as demoTalkNet.py evaluate_network
MIN_TRACK_FRAMES_25FPS = 10  # demoTalkNet.py's --minTrack default; shorter tracks are unreliable


def ensure_repo_on_path():
    if not os.path.isdir(TALKNET_REPO):
        raise SystemExit(
            f"TalkNet-ASD repo not found at {TALKNET_REPO}.\n"
            f"Run: git clone https://github.com/TaoRuijie/TalkNet-ASD {TALKNET_REPO}"
        )
    sys.path.insert(0, TALKNET_REPO)


def ensure_pretrained_model():
    model_path = os.path.join(TALKNET_REPO, PRETRAIN_FILENAME)
    if os.path.isfile(model_path):
        return model_path
    print(f"Pretrained model not found, downloading via gdown (id={PRETRAIN_GDRIVE_ID})...")
    cmd = f"gdown --id {PRETRAIN_GDRIVE_ID} -O {model_path}"
    ret = subprocess.call(cmd, shell=True)
    if ret != 0 or not os.path.isfile(model_path):
        raise SystemExit(
            "gdown failed. If Google Drive's automated-download limit was hit, "
            f"download manually from the file id {PRETRAIN_GDRIVE_ID} and place at {model_path}"
        )
    return model_path


def reencode_and_extract_frames(video_path, work_dir):
    """Matches demoTalkNet.py's own preprocessing exactly: re-encode to 25fps,
    extract mono 16kHz audio, dump every frame as a jpg."""
    os.makedirs(work_dir, exist_ok=True)
    video_25fps = os.path.join(work_dir, "video_25fps.avi")
    audio_wav = os.path.join(work_dir, "audio.wav")
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.isfile(video_25fps):
        subprocess.call(
            f"ffmpeg -y -i {video_path} -qscale:v 2 -async 1 -r {TARGET_FPS} {video_25fps} -loglevel panic",
            shell=True)
    if not os.path.isfile(audio_wav):
        subprocess.call(
            f"ffmpeg -y -i {video_25fps} -qscale:a 0 -ac 1 -vn -ar 16000 {audio_wav} -loglevel panic",
            shell=True)
    if not glob.glob(os.path.join(frames_dir, "*.jpg")):
        subprocess.call(
            f"ffmpeg -y -i {video_25fps} -qscale:v 2 -f image2 {os.path.join(frames_dir, '%06d.jpg')} -loglevel panic",
            shell=True)
    return video_25fps, audio_wav, frames_dir


def build_dense_tracks(face_tracks_json, native_fps_original):
    """Converts our sparse (5fps, per-shot) MTCNN tracks into demoTalkNet.py's
    dense per-25fps-frame track format: {'frame': int array, 'bbox': Nx4 array}.
    Gaps are linearly interpolated (same approach as the original track_shot),
    and box center/size are median-filtered (same as the original crop_video)
    to smooth out MTCNN jitter between sparse detections.
    """
    dense_tracks = []  # each: {"shot_id", "track_id", "frame": np.array, "bbox": np.array Nx4}
    for shot in face_tracks_json["shots"]:
        for tr in shot["tracks"]:
            dets = tr["detections"]
            if len(dets) < 2:
                continue
            ts = np.array([d["t"] for d in dets])
            boxes_xywh = np.array([d["box"] for d in dets], dtype=np.float64)  # x,y,w,h
            boxes_xyxy = np.stack([
                boxes_xywh[:, 0], boxes_xywh[:, 1],
                boxes_xywh[:, 0] + boxes_xywh[:, 2], boxes_xywh[:, 1] + boxes_xywh[:, 3],
            ], axis=1)

            frame_start = int(round(ts[0] * TARGET_FPS))
            frame_end = int(round(ts[-1] * TARGET_FPS))
            if frame_end - frame_start + 1 < MIN_TRACK_FRAMES_25FPS:
                continue
            frame_idx = np.arange(frame_start, frame_end + 1)
            frame_t = frame_idx / TARGET_FPS

            bbox_dense = np.stack([
                np.interp(frame_t, ts, boxes_xyxy[:, c]) for c in range(4)
            ], axis=1)

            dense_tracks.append({
                "shot_id": shot["shot_id"],
                "track_id": tr["track_id"],
                "frame": frame_idx,
                "bbox": bbox_dense,
            })
    return dense_tracks


def crop_track(frames_dir, audio_wav_full, track, out_prefix):
    """Ported from demoTalkNet.py's crop_video(). Produces a 224x224 face-centered
    .avi + matching .wav for one track."""
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    bbox = track["bbox"]
    s = np.maximum(bbox[:, 3] - bbox[:, 1], bbox[:, 2] - bbox[:, 0]) / 2
    x = (bbox[:, 0] + bbox[:, 2]) / 2
    y = (bbox[:, 1] + bbox[:, 3]) / 2
    if len(s) >= MEDFILT_KERNEL:
        s = signal.medfilt(s, kernel_size=MEDFILT_KERNEL)
        x = signal.medfilt(x, kernel_size=MEDFILT_KERNEL)
        y = signal.medfilt(y, kernel_size=MEDFILT_KERNEL)

    avi_path = out_prefix + ".avi"
    vout = cv2.VideoWriter(out_prefix + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), TARGET_FPS, (224, 224))
    for i, frame_num in enumerate(track["frame"]):
        if frame_num < 0 or frame_num >= len(flist):
            continue
        image = cv2.imread(flist[frame_num])
        if image is None:
            continue
        cs = CROP_SCALE
        bs = s[i]
        bsi = int(bs * (1 + 2 * cs))
        padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = y[i] + bsi, x[i] + bsi
        face = padded[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        if face.size == 0:
            continue
        vout.write(cv2.resize(face, (224, 224)))
    vout.release()

    t0, t1 = track["frame"][0] / TARGET_FPS, (track["frame"][-1] + 1) / TARGET_FPS
    wav_path = out_prefix + ".wav"
    subprocess.call(
        f"ffmpeg -y -i {audio_wav_full} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "
        f"-ss {t0:.3f} -to {t1:.3f} {wav_path} -loglevel panic", shell=True)
    subprocess.call(
        f"ffmpeg -y -i {out_prefix}t.avi -i {wav_path} -c:v copy -c:a copy {avi_path} -loglevel panic",
        shell=True)
    os.remove(out_prefix + "t.avi")
    return avi_path, wav_path


def score_track(talknet_model, avi_path, wav_path):
    """Ported from demoTalkNet.py's evaluate_network(): multi-duration-ensemble
    scoring of one track's cropped clip. Returns one score per 25fps frame,
    where >= 0 conventionally means "speaking" (per the original repo's own
    visualization threshold) -- this is a raw class-1 logit from the model's
    AV head, not a calibrated probability."""
    import torch
    import python_speech_features

    _, audio = wavfile.read(wav_path)
    audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

    video = cv2.VideoCapture(avi_path)
    video_feature = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face[56:168, 56:168]  # 112x112 center crop, same as original
        video_feature.append(face)
    video.release()
    video_feature = np.array(video_feature)

    length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
    if length <= 0:
        return np.array([])
    audio_feature = audio_feature[:int(round(length * 100)), :]
    video_feature = video_feature[:int(round(length * 25)), :, :]

    all_scores = []
    for duration in DURATION_SET:
        batch_size = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batch_size):
                input_a = torch.FloatTensor(
                    audio_feature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                input_v = torch.FloatTensor(
                    video_feature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                embed_a = talknet_model.model.forward_audio_frontend(input_a)
                embed_v = talknet_model.model.forward_visual_frontend(input_v)
                embed_a, embed_v = talknet_model.model.forward_cross_attention(embed_a, embed_v)
                out = talknet_model.model.forward_audio_visual_backend(embed_a, embed_v)
                score = talknet_model.lossAV.forward(out, labels=None)
                scores.extend(score)
        all_scores.append(scores)
    return np.round(np.mean(np.array(all_scores), axis=0), 3).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    ensure_repo_on_path()
    from talkNet import talkNet  # noqa: E402  (needs sys.path modified first)

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    shots_path = os.path.join(args.data_dir, "shots", f"{args.video_id}_shots.json")
    tracks_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_tracks.json")
    transcript_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_transcript.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution_talknet.json")
    work_dir = os.path.join(args.data_dir, "talknet_work", args.video_id)

    with open(tracks_path, encoding="utf-8") as f:
        face_tracks_json = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)["transcript"]

    cap = cv2.VideoCapture(video_path)
    native_fps_original = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("Re-encoding to 25fps + extracting frames/audio (matches TalkNet-ASD's own preprocessing)...")
    _, audio_wav, frames_dir = reencode_and_extract_frames(video_path, work_dir)

    print("Converting face tracks to dense 25fps format...")
    dense_tracks = build_dense_tracks(face_tracks_json, native_fps_original)
    print(f"  {len(dense_tracks)} track(s) usable (>= {MIN_TRACK_FRAMES_25FPS} frames at {TARGET_FPS}fps)")

    print("Loading pretrained TalkNet-ASD model...")
    model_path = ensure_pretrained_model()
    talknet_model = talkNet()
    talknet_model.loadParameters(model_path)
    talknet_model.eval()

    crops_dir = os.path.join(work_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    track_scores = []  # list of {shot_id, track_id, frame(np array), score(np array)}
    for i, tr in enumerate(dense_tracks):
        prefix = os.path.join(crops_dir, f"shot{tr['shot_id']:03d}_track{tr['track_id']:03d}")
        print(f"  [{i+1}/{len(dense_tracks)}] scoring shot{tr['shot_id']}_track{tr['track_id']} "
              f"({len(tr['frame'])} frames)...")
        avi_path, wav_path = crop_track(frames_dir, audio_wav, tr, prefix)
        scores = score_track(talknet_model, avi_path, wav_path)
        n = min(len(scores), len(tr["frame"]))
        track_scores.append({
            "shot_id": tr["shot_id"], "track_id": tr["track_id"],
            "frame": tr["frame"][:n], "score": scores[:n],
        })

    # Map each track's per-frame scores onto caption segments by time window.
    results = []
    for seg in transcript:
        seg_f0 = int(round(seg["start_sec"] * TARGET_FPS))
        seg_f1 = int(round(seg["end_sec"] * TARGET_FPS))
        candidates = []
        for tsc in track_scores:
            mask = (tsc["frame"] >= seg_f0) & (tsc["frame"] <= seg_f1)
            if mask.sum() == 0:
                continue
            candidates.append({
                "shot_id": tsc["shot_id"], "track_id": tsc["track_id"],
                "mean_score": round(float(np.mean(tsc["score"][mask])), 3),
                "n_frames": int(mask.sum()),
            })
        candidates.sort(key=lambda c: c["mean_score"], reverse=True)

        if candidates and candidates[0]["mean_score"] >= 0:
            speaker_id = f"shot{candidates[0]['shot_id']}_track{candidates[0]['track_id']}"
            status = "assigned"
        elif candidates:
            speaker_id = f"shot{candidates[0]['shot_id']}_track{candidates[0]['track_id']}"
            status = "assigned_low_confidence"  # best available, but TalkNet itself scored it as more "not speaking"
        else:
            speaker_id, status = None, "uncertain_no_confident_face"

        results.append({
            "segment_id": seg["segment_id"], "start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
            "text_pt": seg["text_pt"], "assigned_speaker": speaker_id, "status": status,
            "candidates": candidates,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "attribution": results}, f, indent=2, ensure_ascii=False)

    print(f"\nTalkNet-ASD speaker attribution -> {out_path}\n")
    for r in results:
        cand_str = ", ".join(f"{c['shot_id']}/{c['track_id']}:{c['mean_score']:+.2f}" for c in r["candidates"])
        print(f"  [{r['start_sec']:6.2f}-{r['end_sec']:6.2f}s] {r['status']:24s} "
              f"speaker={r['assigned_speaker']}  ({cand_str})  \"{r['text_pt'][:40]}\"")


if __name__ == "__main__":
    main()
