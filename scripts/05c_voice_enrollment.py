"""
Step 5c: voice enrollment -- build a per-identity voice profile from segments
where face-based attribution (05/05b) was confident, so step 5d can later
resolve segments where no face is visible (or where face-based signals
conflict) using voice alone.

WHY THIS APPROACH: this pipeline's persistent identities (04b/04c) already
solve "the host looks different across shots" for FACE. The remaining
failure mode -- flagged directly from real output on this project's own
video -- is speech continuing (or starting) off-screen: a segment where
TalkNet gave two candidates strongly positive scores at once (e.g. +2.40
and +1.79 in the same window) is the signature of two different people's
speech blending into one averaged face-based score, not a confident single
answer. Voice doesn't have that failure mode -- it only needs audio, so it
can resolve exactly the segments face-based methods structurally can't.

MODEL: resemblyzer (https://github.com/resemble-ai/Resemblyzer), a
lightweight LSTM-based speaker encoder. Chosen specifically because its
pretrained weights (17MB) ship INSIDE the pip package itself -- no download
step at all, not even a GitHub release like keras-facenet needed. This
sandbox could not actually run it end-to-end: `pip install resemblyzer`
pulls in torch, and this particular sandbox's torch install has a broken/
incomplete CUDA library chain (confirmed: the installed nvidia-cuda-runtime
package is missing libcudart.so.13 itself) combined with very little disk
headroom to fix it. This is a sandbox-specific problem, not a code problem
-- your server already runs torch/CUDA successfully for TalkNet-ASD. The
API used here (encoder.embed_speaker(), encoder.embed_utterance()) was
verified by reading resemblyzer's actual source directly (voice_encoder.py,
audio.py), not from memory, for the same reason TalkNet's preprocessing was
ported from its real source rather than guessed. Everything EXCEPT the
actual embedding computation itself was tested in this sandbox with mock
embeddings standing in for the real model -- audio slicing, the
enrollment-eligibility logic, and all the JSON plumbing.

CONFIDENCE THRESHOLDS ARE NOT YET CALIBRATED like the rest of this
pipeline's thresholds were (OCR confidence, face-clustering distance) --
those were set by measuring real same/different examples from actual
project footage. That kind of validation isn't possible here since the
model can't run in this sandbox. Treat MIN_ENROLLMENT_DURATION_SEC and the
enrollment-eligibility rule below as starting points to sanity-check
against your own results, not as measured constants.

Usage:
    python3 05c_voice_enrollment.py <video_id>

Reads:  data/audio/<video_id>.wav
        data/episodes/<video_id>_transcript.json
        data/episodes/<video_id>_speaker_attribution.json         (heuristic, optional)
        data/episodes/<video_id>_speaker_attribution_talknet.json (TalkNet, optional -- preferred if both present)
Writes: data/episodes/<video_id>_voice_profiles.npz
        data/episodes/<video_id>_voice_enrollment_log.json  (which segments enrolled which identity, for review)
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

MIN_ENROLLMENT_DURATION_SEC = 3.0  # per identity, across all its confident segments combined
CONFIDENT_STATUSES_TALKNET = {"assigned"}
CONFIDENT_STATUSES_HEURISTIC = {"assigned_single_candidate", "assigned_multi_candidate"}


def load_encoder():
    """Imported lazily so this script's non-model logic can be exercised without
    torch installed at all. See module docstring for why this couldn't be
    executed end-to-end in this sandbox."""
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def load_attribution(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {r["segment_id"]: r for r in json.load(f)["attribution"]}


def select_enrollment_segments(transcript, heur_attr, talk_attr):
    """Returns {segment_id: identity} for segments confident enough to enroll.

    Preference order:
      1. Both methods available and agree on the same identity, with at
         least one of them not flagged low-confidence -- the strongest
         signal this pipeline can currently produce.
      2. Only one method available (or they disagree) -- fall back to that
         method's own "confident" statuses. Prefers TalkNet over the
         heuristic when both exist but disagree, consistent with every
         other tie-break in this pipeline (script 06's comparisons showed
         TalkNet consistently more internally consistent on this project's
         real footage).
    """
    selected = {}
    for seg in transcript:
        sid = seg["segment_id"]
        h = heur_attr.get(sid) if heur_attr else None
        t = talk_attr.get(sid) if talk_attr else None

        if h and t and h["assigned_speaker"] and h["assigned_speaker"] == t["assigned_speaker"]:
            not_both_low_conf = not (h["status"].endswith("low_conf") and t["status"] == "assigned_low_confidence")
            if not_both_low_conf:
                selected[sid] = t["assigned_speaker"]
                continue

        if t and t["status"] in CONFIDENT_STATUSES_TALKNET:
            selected[sid] = t["assigned_speaker"]
        elif h and h["status"] in CONFIDENT_STATUSES_HEURISTIC:
            selected[sid] = h["assigned_speaker"]
        # else: not confident enough by either method -- not used for enrollment.
    return selected


def extract_segment_audio(wav_data, sr, start_sec, end_sec):
    i0, i1 = int(start_sec * sr), int(end_sec * sr)
    return wav_data[max(0, i0):min(len(wav_data), i1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--min-duration-sec", type=float, default=MIN_ENROLLMENT_DURATION_SEC)
    args = parser.parse_args()

    wav_path = os.path.join(args.data_dir, "audio", f"{args.video_id}.wav")
    transcript_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_transcript.json")
    heur_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution.json")
    talk_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution_talknet.json")
    profiles_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_voice_profiles.npz")
    log_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_voice_enrollment_log.json")

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)["transcript"]
    heur_attr = load_attribution(heur_path)
    talk_attr = load_attribution(talk_path)
    if not heur_attr and not talk_attr:
        raise SystemExit("No attribution file found (heuristic or TalkNet) -- run 05/05b first.")

    selected = select_enrollment_segments(transcript, heur_attr, talk_attr)
    print(f"{len(selected)}/{len(transcript)} segments confident enough to enroll")

    seg_by_id = {s["segment_id"]: s for s in transcript}
    by_identity = {}
    for sid, identity in selected.items():
        by_identity.setdefault(identity, []).append(seg_by_id[sid])

    wav_data, sr = sf.read(wav_path)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)  # downmix to mono if needed

    print("Loading speaker encoder...")
    encoder = load_encoder()
    from resemblyzer import preprocess_wav

    profiles = {}
    log = {}
    for identity, segs in by_identity.items():
        total_dur = sum(s["end_sec"] - s["start_sec"] for s in segs)
        if total_dur < args.min_duration_sec:
            print(f"  {identity}: skipping enrollment, only {total_dur:.1f}s of confident audio "
                  f"(need >= {args.min_duration_sec}s)")
            log[identity] = {"enrolled": False, "total_duration_sec": round(total_dur, 2),
                              "n_segments": len(segs), "reason": "insufficient_duration"}
            continue

        wavs = []
        for s in segs:
            raw = extract_segment_audio(wav_data, sr, s["start_sec"], s["end_sec"])
            if len(raw) == 0:
                continue
            wavs.append(preprocess_wav(raw.astype(np.float32), source_sr=sr))
        if not wavs:
            log[identity] = {"enrolled": False, "total_duration_sec": round(total_dur, 2),
                              "n_segments": len(segs), "reason": "no_usable_audio_after_preprocessing"}
            continue

        profile = encoder.embed_speaker(wavs)
        profiles[identity] = profile
        print(f"  {identity}: enrolled from {len(segs)} segment(s), {total_dur:.1f}s total audio")
        log[identity] = {"enrolled": True, "total_duration_sec": round(total_dur, 2),
                          "n_segments": len(segs),
                          "segment_ids": [s["segment_id"] for s in segs]}

    if not profiles:
        raise SystemExit("No identity had enough confident audio to enroll a voice profile. "
                          f"Lower --min-duration-sec (currently {args.min_duration_sec}) or "
                          "check that 05/05b produced confident assignments.")

    np.savez(profiles_path, **profiles)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\nVoice profiles ({len(profiles)} identities) -> {profiles_path}")
    print(f"Enrollment log -> {log_path}")


if __name__ == "__main__":
    main()
