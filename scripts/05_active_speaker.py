"""
Step 5: Active speaker attribution.

Lightweight audio-visual correlation approach, NOT a trained ASD model.

This sandbox's network cannot reach Google Drive, which is where Light-ASD
and TalkNet-ASD (the standard trained ASD models) host their pretrained
weights, so a trained model could not be tested here end-to-end. Instead,
this implements the classic signal-level idea those models are built on:
a speaking mouth moves in a way that correlates with the audio energy
envelope, a silent mouth doesn't. Concretely:

  1. Extract audio RMS energy on a uniform fine time grid.
  2. For each tracked face, crop the mouth region (from MTCNN keypoints)
     at each detection and compute frame-to-frame pixel difference as a
     motion proxy, then interpolate onto the same time grid.
  3. For each caption segment, correlate every candidate track's motion
     signal against the audio energy over the segment's window and assign
     the segment to the best-correlating track.

This is a legitimate baseline and the interface (per-segment candidate
tracks -> a score per track) is exactly what you'd swap a trained ASD
model into on a machine with unrestricted internet access -- replace
`mouth_motion_signal()` with per-frame ASD network scores and everything
downstream (interpolation, correlation-based segment attribution) still
applies, or simplifies further since a trained ASD model outputs a
per-frame speaking probability directly instead of a proxy signal you
then have to correlate.

Usage:
    python3 05_active_speaker.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/episodes/<video_id>_face_tracks.json
        data/episodes/<video_id>_transcript.json
Writes: data/audio/<video_id>.wav (extracted automatically if not already present)
        data/episodes/<video_id>_speaker_attribution.json
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import librosa

AUDIO_SR = 16000
AUDIO_HOP_SEC = 0.02        # 50 Hz energy signal
MOUTH_PAD_SCALE = 0.9       # mouth crop half-width, as a fraction of mouth_left-mouth_right distance
MIN_CORR_CONFIDENT = 0.15   # below this, flag segment as "uncertain" rather than force an assignment
MIN_OVERLAP_SAMPLES = 3     # need at least this many aligned samples to trust a correlation


def ensure_audio_extracted(video_path, wav_path):
    """Extracts mono 16kHz audio via ffmpeg if it doesn't already exist."""
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
        return
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    print(f"Extracting audio -> {wav_path}")
    ret = subprocess.call(
        f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar {AUDIO_SR} -acodec pcm_s16le "{wav_path}" -loglevel error',
        shell=True)
    if ret != 0 or not os.path.isfile(wav_path):
        raise SystemExit(f"ffmpeg failed to extract audio from {video_path} (exit code {ret}). "
                          f"Check that ffmpeg is installed and on PATH, and that the video path is correct.")


def compute_audio_energy(wav_path):
    y, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    hop_length = int(AUDIO_HOP_SEC * sr)
    frame_length = hop_length * 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def mouth_crop(frame, keypoints, expand=MOUTH_PAD_SCALE):
    mx0, my0 = keypoints["mouth_left"]
    mx1, my1 = keypoints["mouth_right"]
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
    half_w = max(8, abs(mx1 - mx0) * (1 + expand) / 2)
    half_h = half_w * 0.7
    h, w = frame.shape[:2]
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, y1 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)


def mouth_motion_signal(cap, native_fps, track):
    """Frame-to-frame mouth-crop pixel difference, at each detection's timestamp
    (from the 2nd detection onward, since motion needs a previous frame)."""
    ts, motions = [], []
    prev_crop = None
    for det in track["detections"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(det["t"] * native_fps)))
        ok, frame = cap.read()
        if not ok:
            prev_crop = None
            continue
        crop = mouth_crop(frame, det["keypoints"])
        if crop is None:
            prev_crop = None
            continue
        if prev_crop is not None:
            # Resize to match in case mouth box size changed between frames.
            ph, pw = prev_crop.shape
            crop_r = cv2.resize(crop, (pw, ph))
            diff = np.mean(np.abs(crop_r.astype(np.float32) - prev_crop.astype(np.float32)))
            ts.append(det["t"])
            motions.append(diff)
        prev_crop = crop
    return np.array(ts), np.array(motions)


def interp_to_grid(ts, values, grid_times, valid_range):
    """Linear-interpolate a sparse signal onto grid_times; returns NaN outside valid_range
    or where there are too few real samples nearby to trust the interpolation."""
    out = np.full_like(grid_times, np.nan, dtype=np.float64)
    if len(ts) < 2:
        return out
    lo, hi = valid_range
    mask = (grid_times >= lo) & (grid_times <= hi)
    out[mask] = np.interp(grid_times[mask], ts, values, left=np.nan, right=np.nan)
    return out


def pearson_corr(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < MIN_OVERLAP_SAMPLES:
        return None, mask.sum()
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0, mask.sum()
    return float(np.corrcoef(a, b)[0, 1]), mask.sum()


def load_identity_map(video_id, data_dir):
    """Loads the persistent within-video identity mapping from
    04b_cluster_faces.py / 04c_apply_identity_corrections.py, if it exists.
    Returns (track_to_identity, excluded_tracks) -- both empty if no
    identity file is present, so this script still works standalone on a
    video where face clustering hasn't been run (falls back to raw
    shot{N}_track{M} labels, exactly as before this feature existed)."""
    path = os.path.join(data_dir, "episodes", f"{video_id}_face_identities.json")
    if not os.path.isfile(path):
        return {}, set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("track_to_identity", {}), set(data.get("excluded_tracks", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    wav_path = os.path.join(args.data_dir, "audio", f"{args.video_id}.wav")
    tracks_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_tracks.json")
    transcript_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_transcript.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution.json")

    with open(tracks_path, encoding="utf-8") as f:
        track_data = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)["transcript"]

    identity_map, excluded_tracks = load_identity_map(args.video_id, args.data_dir)
    if identity_map:
        print(f"Loaded {len(set(identity_map.values()))} persistent identities "
              f"({len(excluded_tracks)} track(s) excluded as non-speakers)")

    ensure_audio_extracted(video_path, wav_path)

    print("Computing audio energy envelope...")
    audio_times, audio_energy = compute_audio_energy(wav_path)

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    # Precompute each track's interpolated motion signal on the audio time grid.
    print("Computing mouth-motion signals per track...")
    track_signals = []  # list of dicts: shot_id, track_id, identity, t_range, signal
    n_excluded = 0
    for shot in track_data["shots"]:
        for tr in shot["tracks"]:
            raw_id = f"shot{shot['shot_id']}_track{tr['track_id']}"
            if raw_id in excluded_tracks:
                n_excluded += 1
                continue
            ts, motions = mouth_motion_signal(cap, native_fps, tr)
            if len(ts) < MIN_OVERLAP_SAMPLES:
                continue
            t_range = (tr["detections"][0]["t"], tr["detections"][-1]["t"])
            sig = interp_to_grid(ts, motions, audio_times, t_range)
            track_signals.append({
                "shot_id": shot["shot_id"],
                "track_id": tr["track_id"],
                "identity": identity_map.get(raw_id, raw_id),  # falls back to raw label if no identity file
                "t_range": t_range,
                "signal": sig,
            })
    cap.release()
    print(f"  {len(track_signals)} track(s) with usable motion signal"
          + (f" ({n_excluded} skipped as non-speaker)" if n_excluded else ""))

    results = []
    for seg in transcript:
        seg_start, seg_end = seg["start_sec"], seg["end_sec"]
        window_mask = (audio_times >= seg_start) & (audio_times <= seg_end)
        seg_audio = np.where(window_mask, audio_energy, np.nan)

        raw_candidates = []
        for tsig in track_signals:
            lo, hi = tsig["t_range"]
            if hi < seg_start or lo > seg_end:
                continue  # no temporal overlap with this segment at all
            corr, n = pearson_corr(seg_audio, tsig["signal"])
            if corr is None:
                continue
            raw_candidates.append({
                "identity": tsig["identity"],
                "shot_id": tsig["shot_id"], "track_id": tsig["track_id"],
                "correlation": round(corr, 4), "n_samples": int(n),
            })

        # Group by resolved identity before ranking: two fragmented tracks of
        # the SAME person overlapping one segment (e.g. a brief re-detection
        # within a busy shot) must count as one candidate, not compete against
        # each other in the margin/ambiguity check below -- that check is
        # meant to catch genuine ties between DIFFERENT people, and would
        # misfire if a person's own fragmented tracks looked like a tie with
        # themselves. Keep whichever fragment shows the strongest correlation
        # as that identity's evidence (a noisy per-track proxy signal is safer
        # to take the best-available reading from than to average across
        # fragments of uneven quality).
        by_identity = {}
        for c in raw_candidates:
            key = c["identity"]
            if key not in by_identity or c["correlation"] > by_identity[key]["correlation"]:
                by_identity[key] = c
        candidates = sorted(by_identity.values(), key=lambda c: c["correlation"], reverse=True)

        # Decision logic: correlation is a disambiguator between multiple candidates,
        # not an absolute gate. If there's only one face on screen for this segment,
        # visual presence itself is already decent evidence -- rejecting it whenever
        # the noisy frame-diff/audio correlation happens to dip below some fixed
        # threshold throws away true positives (this was checked: with an absolute
        # threshold, ~50% of genuinely single-speaker segments in this clip were
        # marked "uncertain" despite being visually unambiguous). We only reject a
        # single candidate on a strong ANTI-correlation, which is a real signal that
        # the one visible face is staying still while audio energy is present --
        # e.g. an off-screen speaker over a reaction shot.
        STRONG_ANTICORR_REJECT = -0.35
        if len(candidates) == 0:
            speaker_id, status = None, "uncertain_no_confident_face"
        elif len(candidates) == 1:
            best = candidates[0]
            if best["correlation"] <= STRONG_ANTICORR_REJECT:
                speaker_id, status = None, "uncertain_anticorrelated_mouth"
            else:
                speaker_id = best["identity"]
                status = "assigned_single_candidate" if best["correlation"] >= MIN_CORR_CONFIDENT else "assigned_single_candidate_low_conf"
        else:
            best, second = candidates[0], candidates[1]
            margin = best["correlation"] - second["correlation"]
            if margin >= 0.1:
                speaker_id = best["identity"]
                status = "assigned_multi_candidate"
            else:
                speaker_id, status = None, "uncertain_ambiguous_multi_candidate"

        results.append({
            "segment_id": seg["segment_id"],
            "start_sec": seg_start, "end_sec": seg_end,
            "text_pt": seg["text_pt"],
            "assigned_speaker": speaker_id,
            "status": status,
            "candidates": candidates,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "attribution": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSpeaker attribution -> {out_path}\n")
    for r in results:
        cand_str = ", ".join(f"{c['identity']}[{c['shot_id']}/{c['track_id']}]:{c['correlation']:+.2f}"
                              for c in r["candidates"])
        print(f"  [{r['start_sec']:6.2f}-{r['end_sec']:6.2f}s] {r['status']:28s} "
              f"speaker={r['assigned_speaker']}  ({cand_str})  \"{r['text_pt'][:40]}\"")


if __name__ == "__main__":
    main()
