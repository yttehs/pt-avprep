"""
Step 4b: cluster face tracks into persistent within-video identities.

04_track_faces.py resets tracking at every shot boundary (correctly -- a cut
can swap who's on screen), which means the same physical person (typically
the host/interviewer, who reappears across many shots) gets a different
track_id every time they're re-detected. This stage links tracks belonging
to the same person back together using face-embedding similarity, so
downstream speaker attribution can use one persistent identity per person
instead of disjoint per-shot track ids.

Scope: within a single video only -- no cross-video identity gallery.

APPROACH, AND WHY THE THRESHOLD IS WHAT IT IS: validated against real face
crops from this project's own footage (the host, appearing in 4 different
shots, vs. two clearly different interviewees) before picking any numbers:
  - Averaging embeddings over ~8 frames per track substantially stabilizes
    same-person similarity (0.57-0.72 on single frames -> 0.71-0.86 averaged)
    while barely moving different-person similarity (stayed under 0.40
    throughout) -- so multi-frame averaging is the default, not optional.
  - One track (steep, unbroken side-profile throughout its entire duration,
    confirmed by inspecting all its frames -- not fixable by picking a
    different frame from the same track) never exceeded ~0.39 similarity to
    the other three genuine same-person crops, i.e. BELOW the ~0.40 ceiling
    observed between different people. A threshold that's forgiving enough
    to catch this track would risk merging different people elsewhere.
  - Given that asymmetry -- a missed merge just leaves two correct identities
    instead of one (annoying, safe) but a wrong merge conflates two different
    people's speech (a real ground-truth error) -- the threshold is set
    conservatively. Tracks that don't clearly clear it stay as their own
    singleton identity rather than being forced into a guess, and their
    similarity to the nearest other identity is reported so a human reviewer
    can see exactly which singletons are worth a manual look (e.g. this
    project's own shot8_track1 would show up flagged at ~0.39 -- a genuine
    "possibly the host, but the source footage doesn't give the model enough
    to say so" case, not a pipeline bug).
  - Average-linkage hierarchical clustering on 1-cosine_similarity was stable
    across thresholds 0.45-0.6 on this test data; single-linkage was NOT
    (it chained an unrelated person in at threshold 0.6) and is avoided here.

  KNOWN LIMITATION (found on the full video, not just the small test clip):
  average-linkage's cluster-average criterion gets more conservative as a
  cluster grows large and diverse -- confirmed concretely when the host's
  cluster grew to 8 members across a full 6-minute video: a track with 0.737
  direct similarity to a specific existing member (well above the 0.5
  threshold, and independently confirmed correct on the smaller test) still
  didn't merge, because its AVERAGE similarity across all 8 members dropped
  below threshold. A "fall back to nearest-single-member similarity" fix was
  considered and rejected: checked against this project's own real output,
  a confirmed-correct merge (0.531) and a confirmed-wrong merge (0.545)
  overlapped in the same similarity range, so no single cutoff separates
  them. See save_borderline_review_sheet() -- this is handled as a human
  review step instead, not an automatic threshold.

Usage:
    python3 04b_cluster_faces.py <video_id>

Reads:  data/raw/<video_id>.mp4
        data/episodes/<video_id>_face_tracks.json
Writes: data/episodes/<video_id>_face_identities.json
        outputs/<video_id>_identity_similarity.png  (debug heatmap)
        outputs/<video_id>_borderline_review.png  (visual merge-confirmation sheet)
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

FRAMES_PER_TRACK = 8       # embeddings are averaged over up to this many frames
CROP_PAD_FRAC = 0.2        # padding around the raw MTCNN box before embedding
DISTANCE_THRESHOLD = 0.5   # 1 - cosine_similarity; validated stable over 0.45-0.6
LINKAGE_METHOD = "average"  # NOT "single" -- see module docstring
BORDERLINE_REVIEW_MIN_SIM = 0.35  # deliberately generous -- see save_borderline_review_sheet


def load_facenet(gpu_mem_mb=None):
    """Imported lazily so this script's non-model logic (grouping, clustering)
    can be exercised/tested without pulling in TensorFlow.

    GPU handling: if requirements-step4-face.txt's `tensorflow-cpu` is what's
    actually installed, this is moot -- that package has no GPU kernels at
    all, so TensorFlow can only ever run on system RAM regardless of what's
    configured here. But "as long as the right package got installed" is an
    assumption, not a guarantee (e.g. some other dependency could have pulled
    in full `tensorflow` instead), so this enforces it explicitly rather than
    relying on that silently being true:
      - gpu_mem_mb=None (default): hide the GPU from TensorFlow entirely via
        CUDA_VISIBLE_DEVICES, the same way regardless of which tensorflow
        variant is actually present. Must happen before `import tensorflow`
        runs anywhere in the process, which is why it's here rather than at
        module level (this function's import statement is the first place
        that happens).
      - gpu_mem_mb=<N>: allow GPU use but hard-cap it to N MB via TensorFlow's
        own logical device configuration, TF's equivalent of the
        torch.cuda.set_per_process_memory_fraction cap used for TalkNet-ASD.
    """
    if gpu_mem_mb is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from keras_facenet import FaceNet
    import tensorflow as tf

    if gpu_mem_mb is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_mb)])
            print(f"Capped TensorFlow GPU memory: {gpu_mem_mb}MB on {gpus[0].name}")
        else:
            print("gpu_mem_mb was set, but TensorFlow sees no GPU (likely tensorflow-cpu "
                  "is installed, which has no GPU support at all) -- running on CPU anyway.")

    return FaceNet()


def iter_tracks(face_tracks_json):
    for shot in face_tracks_json["shots"]:
        for tr in shot["tracks"]:
            yield shot["shot_id"], tr


def track_embedding(embedder, cap, native_fps, track, frames_per_track=FRAMES_PER_TRACK):
    dets = track["detections"]
    step = max(1, len(dets) // frames_per_track)
    picked = dets[::step][:frames_per_track]
    crops = []
    for det in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(det["t"] * native_fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, w, h = det["box"]
        pad = int(CROP_PAD_FRAC * max(w, h))
        h_frame, w_frame = frame.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_frame, x + w + pad), min(h_frame, y + h + pad)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (160, 160))
        crops.append(crop)
    if not crops:
        return None
    embs = embedder.embeddings(np.array(crops))
    mean_emb = embs.mean(axis=0)
    return mean_emb / np.linalg.norm(mean_emb)


def cluster_tracks(track_ids, embeddings, threshold=DISTANCE_THRESHOLD, method=LINKAGE_METHOD):
    """Returns (labels, similarity_matrix). labels[i] is an integer cluster id
    for track_ids[i]. Falls back to a single cluster if there's only one track,
    since scipy's linkage needs at least 2 observations."""
    n = len(track_ids)
    M = np.array(embeddings)
    sim = M @ M.T  # embeddings are already L2-normalized
    if n < 2:
        return np.zeros(n, dtype=int), sim
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # guard against tiny negative floating-point noise
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels, sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--outputs-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD)
    parser.add_argument("--gpu-mem-mb", type=int, default=None,
                         help="Allow GPU use, hard-capped to this many MB. Default (unset) "
                              "hides the GPU from TensorFlow entirely, forcing CPU-only "
                              "execution -- the safe default on a shared GPU, and moot anyway "
                              "if tensorflow-cpu (no GPU support at all) is what's installed.")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing face_identities.json even if it has "
                              "manual corrections applied (04c). Use only to intentionally "
                              "redo automatic clustering from scratch.")
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "raw", f"{args.video_id}.mp4")
    tracks_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_tracks.json")
    out_path = os.path.join(args.data_dir, "episodes", f"{args.video_id}_face_identities.json")
    plot_path = os.path.join(args.outputs_dir, f"{args.video_id}_identity_similarity.png")

    if os.path.isfile(out_path) and not args.force:
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        if "excluded_tracks" in existing or existing.get("_manually_corrected"):
            raise SystemExit(
                f"{out_path} already has manual corrections applied (via "
                "04c_apply_identity_corrections.py) -- rerunning this script would "
                "regenerate it from scratch and silently discard those corrections. "
                "Pass --force if you really want to redo automatic clustering from "
                "scratch (e.g. with a different --threshold)."
            )

    with open(tracks_path, encoding="utf-8") as f:
        face_tracks_json = json.load(f)

    print("Loading FaceNet embedding model...")
    embedder = load_facenet(gpu_mem_mb=args.gpu_mem_mb)

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    track_ids, embeddings, first_seen = [], [], []
    for shot_id, tr in iter_tracks(face_tracks_json):
        emb = track_embedding(embedder, cap, native_fps, tr)
        if emb is None:
            print(f"  skipping shot{shot_id}_track{tr['track_id']}: no valid crops")
            continue
        track_ids.append(f"shot{shot_id}_track{tr['track_id']}")
        embeddings.append(emb)
        first_seen.append(tr["detections"][0]["t"])
        print(f"  embedded shot{shot_id}_track{tr['track_id']} "
              f"({len(tr['detections'])} detections)")
    cap.release()

    print(f"\nClustering {len(track_ids)} tracks (distance threshold={args.threshold}, "
          f"linkage={LINKAGE_METHOD})...")
    labels, sim = cluster_tracks(track_ids, embeddings, threshold=args.threshold)

    # Name identities by first-appearance order, not raw cluster label, so
    # "person_0" is always whoever shows up earliest in the video.
    cluster_first_seen = {}
    for i, lbl in enumerate(labels):
        cluster_first_seen[lbl] = min(cluster_first_seen.get(lbl, 1e9), first_seen[i])
    ordered_clusters = sorted(cluster_first_seen, key=cluster_first_seen.get)
    identity_name = {lbl: f"person_{i}" for i, lbl in enumerate(ordered_clusters)}

    identities = {}
    for i, tid in enumerate(track_ids):
        identities.setdefault(identity_name[labels[i]], []).append(tid)

    # For every singleton (or small) cluster, report the nearest track OUTSIDE
    # its own cluster -- this is the human-review signal for cases like this
    # project's own shot8_track1: never forced into a merge, but not silently
    # hidden as "definitely a different person" either.
    nearest_other = {}
    for i, tid in enumerate(track_ids):
        best_j, best_sim = None, -1
        for j in range(len(track_ids)):
            if labels[j] == labels[i]:
                continue
            if sim[i, j] > best_sim:
                best_sim, best_j = sim[i, j], j
        if best_j is not None:
            nearest_other[tid] = {"track": track_ids[best_j],
                                   "identity": identity_name[labels[best_j]],
                                   "similarity": round(float(best_sim), 3)}

    track_to_identity = {tid: identity_name[labels[i]] for i, tid in enumerate(track_ids)}

    output = {
        "video_id": args.video_id,
        "params": {"distance_threshold": args.threshold, "linkage": LINKAGE_METHOD,
                   "frames_per_track": FRAMES_PER_TRACK},
        "identities": [
            {"identity": name, "tracks": tracks, "n_tracks": len(tracks)}
            for name, tracks in identities.items()
        ],
        "track_to_identity": track_to_identity,
        "nearest_other_cluster": nearest_other,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFace identities -> {out_path}\n")
    for name, tracks in identities.items():
        print(f"  {name}: {tracks}")
        if len(tracks) == 1 and tracks[0] in nearest_other:
            no = nearest_other[tracks[0]]
            print(f"    (singleton -- nearest other track: {no['track']} "
                  f"[{no['identity']}], similarity={no['similarity']} -- worth a manual look "
                  f"if that number is uncomfortably high)")

    os.makedirs(args.outputs_dir, exist_ok=True)
    save_similarity_heatmap(track_ids, sim, identities, plot_path)
    print(f"\nSimilarity heatmap -> {plot_path}")

    review_path = os.path.join(args.outputs_dir, f"{args.video_id}_borderline_review.png")
    save_borderline_review_sheet(track_ids, identities, nearest_other, cap_path=video_path,
                                  face_tracks_json=face_tracks_json, out_path=review_path)
    print(f"Borderline-merge review sheet -> {review_path}")


def save_borderline_review_sheet(track_ids, identities, nearest_other, cap_path, face_tracks_json, out_path):
    """For every singleton whose nearest-other-cluster similarity is above
    BORDERLINE_REVIEW_MIN_SIM, renders its face crop side by side with a crop
    from that nearest track, labeled with the similarity score, sorted
    highest-first. This exists because the clustering threshold genuinely
    cannot separate all cases on similarity alone (validated against real
    data: a confirmed-correct merge and a confirmed-wrong merge overlapped in
    the same similarity range -- see the module-level notes), so instead of
    chasing a threshold that doesn't exist, this turns every such case into a
    two-second visual yes/no for a human reviewer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    singleton_ids = {tracks[0] for tracks in identities.values() if len(tracks) == 1}
    candidates = [(tid, info) for tid, info in nearest_other.items()
                  if tid in singleton_ids and info["similarity"] >= BORDERLINE_REVIEW_MIN_SIM]
    candidates.sort(key=lambda x: x[1]["similarity"], reverse=True)

    if not candidates:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No borderline singletons above threshold", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        return

    def track_lookup(tid):
        shot_id = int(tid.split("_")[0].replace("shot", ""))
        track_id = int(tid.split("_")[1].replace("track", ""))
        shot = next(s for s in face_tracks_json["shots"] if s["shot_id"] == shot_id)
        return next(t for t in shot["tracks"] if t["track_id"] == track_id)

    def rep_crop(cap, tid, native_fps):
        tr = track_lookup(tid)
        det = tr["detections"][len(tr["detections"]) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(det["t"] * native_fps)))
        ok, frame = cap.read()
        x, y, w, h = det["box"]
        pad = int(0.25 * max(w, h))
        h_f, w_f = frame.shape[:2]
        crop = frame[max(0, y - pad):min(h_f, y + h + pad), max(0, x - pad):min(w_f, x + w + pad)]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(cap_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    fig, axes = plt.subplots(len(candidates), 2, figsize=(7, 2.4 * len(candidates)))
    if len(candidates) == 1:
        axes = axes.reshape(1, 2)
    for row, (tid, info) in enumerate(candidates):
        for col, (label, other_tid) in enumerate([(tid, tid), (info["identity"], info["track"])]):
            ax = axes[row, col]
            crop = rep_crop(cap, other_tid, native_fps)
            ax.imshow(crop)
            ax.axis("off")
            ax.set_title(f"{other_tid}" + (f"\nsim={info['similarity']}" if col == 1 else ""),
                         fontsize=8)
    cap.release()
    fig.suptitle("Borderline singleton merges -- confirm by eye (highest similarity first)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)


def save_similarity_heatmap(track_ids, sim, identities, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(track_ids) * 0.5), max(5, len(track_ids) * 0.5)))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(track_ids))); ax.set_xticklabels(track_ids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(track_ids))); ax.set_yticklabels(track_ids, fontsize=7)
    ax.set_title("Track embedding cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
