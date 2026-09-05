"""
Step 4c: apply manual corrections to face_identities.json after reviewing
the borderline_review.png sheet from 04b_cluster_faces.py.

Three kinds of correction, matching the three outcomes a human reviewer
finds when looking at that sheet:
  1. "merge" -- two rows are actually the same person; fold the singleton
     track into the target identity.
  2. "exclude" -- the track isn't a real speaker at all (a false-positive
     detection, a non-speaking bystander in the background, etc.); pull it
     out of the identity system entirely rather than forcing it into either
     a merge or its own identity.
  3. Anything not mentioned is left alone -- "confirmed different speaker,
     correctly separate" needs no action, which is why this script only
     takes a small list of exceptions rather than requiring you to
     re-specify the whole identity file.

Usage:
    python3 04c_apply_identity_corrections.py <video_id> <corrections.json>

corrections.json format:
{
  "merge": [
    {"track": "shot8_track1", "target_identity": "person_0"}
  ],
  "exclude": ["shot15_track14", "shot3_track1"]
}

Reads:  data/episodes/<video_id>_face_identities.json
        <corrections.json> (path given on the command line, anywhere)
Writes: data/episodes/<video_id>_face_identities.json  (overwritten)
        data/episodes/<video_id>_face_identities.json.bak  (pre-correction copy)
"""
import argparse
import json
import os
import shutil


def find_identity_of(data, track):
    return data["track_to_identity"].get(track)


def remove_track_from_identity(data, track, identity_name):
    """Removes `track` from `identity_name`'s track list, deleting the
    identity entirely if that was its last track. Returns nothing; mutates
    data["identities"] in place."""
    for entry in data["identities"]:
        if entry["identity"] == identity_name:
            if track in entry["tracks"]:
                entry["tracks"].remove(track)
                entry["n_tracks"] = len(entry["tracks"])
            break
    data["identities"] = [e for e in data["identities"] if e["n_tracks"] > 0]


def add_track_to_identity(data, track, identity_name):
    for entry in data["identities"]:
        if entry["identity"] == identity_name:
            if track not in entry["tracks"]:
                entry["tracks"].append(track)
                entry["n_tracks"] = len(entry["tracks"])
            return
    raise ValueError(f"target_identity '{identity_name}' not found in identities list -- "
                      f"check spelling against the actual face_identities.json")


def prune_stale_nearest_other(data):
    """A track's nearest_other_cluster entry is only meaningful while it's a
    singleton (see 04b's save_borderline_review_sheet) -- once corrections
    merge it into a multi-member identity, or exclude it, that entry is
    stale and would be misleading if left in the file."""
    still_singleton = {e["tracks"][0] for e in data["identities"] if e["n_tracks"] == 1}
    data["nearest_other_cluster"] = {
        tid: info for tid, info in data.get("nearest_other_cluster", {}).items()
        if tid in still_singleton
    }


def apply_corrections(data, corrections):
    log = []

    for item in corrections.get("merge", []):
        track, target = item["track"], item["target_identity"]
        current = find_identity_of(data, track)
        if current is None:
            log.append(f"SKIP merge {track} -> {target}: track not found in track_to_identity")
            continue
        if current == target:
            log.append(f"SKIP merge {track} -> {target}: already in that identity")
            continue
        remove_track_from_identity(data, track, current)
        add_track_to_identity(data, track, target)
        data["track_to_identity"][track] = target
        log.append(f"MERGED {track}: {current} -> {target}")

    excluded = data.setdefault("excluded_tracks", [])
    for track in corrections.get("exclude", []):
        current = find_identity_of(data, track)
        if current is None:
            if track in excluded:
                log.append(f"SKIP exclude {track}: already excluded")
            else:
                log.append(f"SKIP exclude {track}: track not found in track_to_identity")
            continue
        remove_track_from_identity(data, track, current)
        del data["track_to_identity"][track]
        excluded.append(track)
        log.append(f"EXCLUDED {track} (was {current})")

    prune_stale_nearest_other(data)
    data["_manually_corrected"] = True
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("corrections_file")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    identities_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_identities.json")
    backup_path = identities_path + ".bak"

    with open(identities_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(args.corrections_file, encoding="utf-8") as f:
        corrections = json.load(f)

    shutil.copy(identities_path, backup_path)
    print(f"Backed up original -> {backup_path}")

    log = apply_corrections(data, corrections)
    for line in log:
        print(f"  {line}")

    with open(identities_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated -> {identities_path}")

    print("\nCurrent identities:")
    for entry in sorted(data["identities"], key=lambda e: e["identity"]):
        print(f"  {entry['identity']}: {entry['tracks']}")
    if data.get("excluded_tracks"):
        print(f"\nExcluded (not real speakers): {data['excluded_tracks']}")


if __name__ == "__main__":
    main()
