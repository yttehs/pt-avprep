"""
Step 5d: resolve final speaker attribution by combining TalkNet (face) and
voice-profile evidence (05c) into a clean two-tier output: every segment is
either genuinely CONFIDENT (one clear answer, from face, voice, or a
validated override of one by the other) or explicitly NOT_CONFIDENT --
never a forced guess dressed up as an answer.

DECISION PROTOCOL (revised after real validation on this project's video --
see below):
  1. TalkNet confident (status == "assigned") on this segment:
       - If voice has a profile for TalkNet's pick AND a different identity
         scores higher by at least OVERRIDE_MARGIN -> override to voice's
         pick (status "confident_voice_override"). This auto-overrides,
         which is a real change from this script's earlier version (it used
         to only ever flag disagreements for human review, never act on
         them) -- see "WHY AUTO-OVERRIDE NOW" below for why that changed.
       - If TalkNet's pick was never enrolled (too little confident audio
         to build a voice profile -- see 05c), there's no voice evidence to
         check against; keep TalkNet's answer as-is
         ("confident_talknet_no_voice_check").
       - Otherwise voice agrees (or doesn't disagree enough to matter) ->
         keep TalkNet's answer ("confident_talknet").
  2. TalkNet NOT confident (low-confidence, uncertain, or no face at all --
     including genuinely off-screen speech): fall back to voice alone.
       - Confident voice match (>= MIN_VOICE_CONFIDENCE) -> use it
         ("confident_voice_only"). This is the main off-screen-speaker fix.
       - Otherwise -> "not_confident", with no speaker forced. This is
         deliberately not further guessed at; see the review tool for how
         to spot-check these by hand.

WHY AUTO-OVERRIDE NOW, WHEN THE REST OF THIS PIPELINE NEVER AUTO-OVERRIDES
PAST AN UNCALIBRATED THRESHOLD: this one specifically stopped being
speculative. An earlier version of this script only flagged disagreements
for review. On this project's real video, EVERY flagged disagreement was
checked by hand against the actual footage: 5 of 6 were confirmed genuine
host interjections during off-screen speech (the exact failure mode this
whole voice layer exists to catch), and the 6th was a separate but also
confirmed real error (audio bleeding across a shot transition). 6-for-6 on
real spot-checks is a real evidence base, not a guess -- so within a single
video, overriding at this margin is now justified. It is NOT necessarily
proven to generalize to a different video's content/genre without its own
spot-check; a text-content heuristic (e.g. "interviewer questions") was
considered and deliberately rejected for being interview-format-specific
and unlikely to generalize to other kinds of video -- the margin-based
check here uses no genre assumptions at all, which is why it was kept
instead.

CONFIDENCE THRESHOLDS: MIN_VOICE_CONFIDENCE (tier 2, voice-alone) has one
confirmed real spot-check on this project's video: 103.17-105.97s, where
TalkNet returned uncertain_ambiguous_multi_candidate (a wide shot -- host
plus two others, plus someone walking through the background -- multiple
faces and background noise). Voice alone correctly resolved it to the
right speaker. The specific timing is worth noting as a likely generalizable
pattern beyond this one clip: TalkNet had also failed on the speaker's next
segment or two right as she started talking, then recovered -- consistent
with a face-based tracker needing a brief moment to "catch up" to a new
active speaker (camera/tracking settling, mouth-sync stabilizing) right at
a speaker-change boundary, a failure mode voice doesn't share since it
needs no settling time. That's a third distinct face-only weak spot on top
of off-screen speech and crowded/wide shots. Still one confirmed example,
not a large validated sample -- treat 0.6 as directionally right, not
precisely tuned. OVERRIDE_MARGIN (tier 1, override-within-confident-face)
has two confirmed real examples: 0.08 sat just below both confirmed real
overrides' margins (0.088 and 0.10) on this video's data, with zero false
positives found in the 6 flagged cases checked. Still only one video's
worth of validation for both thresholds -- worth re-checking after running
on more footage before trusting the exact numbers across a wider variety
of content.

Usage:
    python3 05d_voice_resolve.py <video_id>

Reads:  data/audio/<video_id>.wav
        data/episodes/<video_id>_transcript.json
        data/episodes/<video_id>_voice_profiles.npz
        data/episodes/<video_id>_speaker_attribution_talknet.json (required -- this
            protocol is TalkNet-confidence-gated specifically, not the heuristic's)
Writes: data/episodes/<video_id>_speaker_attribution_final.json
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

MIN_VOICE_CONFIDENCE = 0.6   # tier 2 (voice-alone) bar -- still an educated guess, not spot-checked yet
OVERRIDE_MARGIN = 0.08       # tier 1 (override-within-confident-face) bar -- backed by 6/6 real spot-checks
                              # on this video (see module docstring); both confirmed real overrides had
                              # margins of 0.088 and 0.10, comfortably above this cutoff.


def load_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def load_attribution(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {r["segment_id"]: r for r in json.load(f)["attribution"]}


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def voice_candidates_for_segment(embed, profiles):
    sims = {identity: cosine_sim(embed, prof) for identity, prof in profiles.items()}
    return sorted(sims.items(), key=lambda kv: kv[1], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--min-voice-confidence", type=float, default=MIN_VOICE_CONFIDENCE)
    parser.add_argument("--override-margin", type=float, default=OVERRIDE_MARGIN)
    args = parser.parse_args()

    wav_path = os.path.join(args.data_dir, "audio", f"{args.video_id}.wav")
    transcript_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_transcript.json")
    profiles_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_voice_profiles.npz")
    talk_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution_talknet.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution_final.json")

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)["transcript"]
    profiles_npz = np.load(profiles_path)
    profiles = {k: profiles_npz[k] for k in profiles_npz.keys()}
    talk_attr = load_attribution(talk_path)
    if not talk_attr:
        raise SystemExit("No TalkNet attribution file found -- this protocol is gated on "
                          "TalkNet's own confidence specifically. Run 05b first.")

    wav_data, sr = sf.read(wav_path)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

    print(f"Loaded {len(profiles)} voice profile(s): {list(profiles.keys())}")
    print("Loading speaker encoder...")
    encoder = load_encoder()
    from resemblyzer import preprocess_wav

    counts = {"confident_talknet": 0, "confident_voice_override": 0,
              "confident_talknet_no_voice_check": 0, "confident_voice_only": 0,
              "not_confident": 0}
    results = []
    for seg in transcript:
        sid = seg["segment_id"]
        t = talk_attr.get(sid)
        talknet_confident = t is not None and t["status"] == "assigned"
        talknet_speaker = t["assigned_speaker"] if t else None

        i0, i1 = int(seg["start_sec"] * sr), int(seg["end_sec"] * sr)
        raw = wav_data[max(0, i0):min(len(wav_data), i1)]
        voice_candidates = []
        if len(raw) > 0 and profiles:
            wav = preprocess_wav(raw.astype(np.float32), source_sr=sr)
            if len(wav) > 0:
                embed = encoder.embed_utterance(wav)
                voice_candidates = voice_candidates_for_segment(embed, profiles)
        voice_sims = dict(voice_candidates)

        if talknet_confident:
            if talknet_speaker in voice_sims:
                face_sim = voice_sims[talknet_speaker]
                best_other = next((c for c in voice_candidates if c[0] != talknet_speaker), None)
                if best_other and (best_other[1] - face_sim) >= args.override_margin:
                    final_speaker, final_status = best_other[0], "confident_voice_override"
                else:
                    final_speaker, final_status = talknet_speaker, "confident_talknet"
            else:
                final_speaker, final_status = talknet_speaker, "confident_talknet_no_voice_check"
        else:
            if voice_candidates and voice_candidates[0][1] >= args.min_voice_confidence:
                final_speaker, final_status = voice_candidates[0][0], "confident_voice_only"
            else:
                final_speaker, final_status = None, "not_confident"
        counts[final_status] += 1

        results.append({
            "segment_id": sid, "start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
            "text_pt": seg["text_pt"],
            "talknet_speaker": talknet_speaker, "talknet_confident": talknet_confident,
            "voice_candidates": [{"identity": i, "similarity": round(s, 3)} for i, s in voice_candidates],
            "final_speaker": final_speaker, "final_status": final_status,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "attribution": results}, f, indent=2, ensure_ascii=False)

    n_total = len(results)
    n_confident = n_total - counts["not_confident"]
    print(f"\nFinal attribution -> {out_path}")
    print(f"  confident (talknet, no voice profile to check): {counts['confident_talknet_no_voice_check']}")
    print(f"  confident (talknet, voice didn't override):     {counts['confident_talknet']}")
    print(f"  confident (voice overrode talknet):              {counts['confident_voice_override']}")
    print(f"  confident (voice only, talknet unsure):          {counts['confident_voice_only']}")
    print(f"  NOT confident:                                   {counts['not_confident']}")
    print(f"  -> {n_confident}/{n_total} segments confidently resolved ({100*n_confident/n_total:.0f}%)")
    print()
    for r in results:
        flag = {"confident_voice_override": "  <-- VOICE OVERRODE TALKNET",
                "confident_voice_only": "  <-- voice-only (off-screen?)",
                "not_confident": "  <-- NOT CONFIDENT"}.get(r["final_status"], "")
        cand_str = ", ".join(f"{c['identity']}:{c['similarity']:+.2f}" for c in r["voice_candidates"][:3])
        print(f"  [{r['start_sec']:6.2f}-{r['end_sec']:6.2f}s] talknet={str(r['talknet_speaker']):10s} "
              f"final={str(r['final_speaker']):10s} ({cand_str}){flag}  \"{r['text_pt'][:30]}\"")


if __name__ == "__main__":
    main()
