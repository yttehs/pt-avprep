"""
Compares the heuristic (05) and TalkNet-ASD (05b) speaker attribution outputs
for the same video, segment by segment. Matches segments by (start_sec,
end_sec) rather than segment_id, since the two are computed independently
but should share the same underlying transcript segmentation.

Usage:
    python3 06_compare_attributions.py <video_id>

Reads:  data/episodes/<video_id>_speaker_attribution.json          (heuristic)
        data/episodes/<video_id>_speaker_attribution_talknet.json  (TalkNet)
Writes: nothing -- prints a summary and the disagreement list to stdout.
"""
import argparse
import json
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["start_sec"], r["end_sec"]): r for r in json.load(f)["attribution"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    heur_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution.json")
    talk_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_speaker_attribution_talknet.json")

    heur = load(heur_path)
    talk = load(talk_path)

    keys = sorted(set(heur) & set(talk))
    missing_heur = set(talk) - set(heur)
    missing_talk = set(heur) - set(talk)
    if missing_heur or missing_talk:
        print(f"WARNING: {len(missing_heur)} segment(s) only in TalkNet output, "
              f"{len(missing_talk)} only in heuristic output (different transcript runs?)\n")

    both_assigned = both_none = agree = disagree = one_abstained = 0
    disagreements = []
    abstentions = []

    for k in keys:
        h, t = heur[k], talk[k]
        h_spk, t_spk = h["assigned_speaker"], t["assigned_speaker"]
        if h_spk is None and t_spk is None:
            both_none += 1
        elif h_spk is None or t_spk is None:
            one_abstained += 1
            abstentions.append((k, h, t))
        else:
            both_assigned += 1
            if h_spk == t_spk:
                agree += 1
            else:
                disagree += 1
                disagreements.append((k, h, t))

    n = len(keys)
    print(f"Compared {n} segments (matched by start/end time)\n")
    print(f"  Both assigned a speaker:      {both_assigned:3d}")
    print(f"    - agreed on who:            {agree:3d}  ({100*agree/max(1,both_assigned):.0f}% of those both assigned)")
    print(f"    - disagreed:                {disagree:3d}")
    print(f"  Only one method assigned:     {one_abstained:3d}  (other abstained/uncertain)")
    print(f"  Both abstained/uncertain:     {both_none:3d}")
    print(f"\n  Overall agreement (agree / all segments): {100*agree/max(1,n):.0f}%")

    if disagreements:
        print(f"\n--- Disagreements ({len(disagreements)}) -- worth spot-checking these against the video ---")
        for k, h, t in disagreements:
            print(f"  [{k[0]:6.2f}-{k[1]:6.2f}s] heuristic={h['assigned_speaker']} ({h['status']})  "
                  f"vs  talknet={t['assigned_speaker']} ({t['status']})  \"{h['text_pt'][:40]}\"")

    if abstentions:
        print(f"\n--- One method abstained, other didn't ({len(abstentions)}) ---")
        for k, h, t in abstentions:
            print(f"  [{k[0]:6.2f}-{k[1]:6.2f}s] heuristic={h['assigned_speaker']} ({h['status']})  "
                  f"vs  talknet={t['assigned_speaker']} ({t['status']})  \"{h['text_pt'][:40]}\"")


if __name__ == "__main__":
    main()
