# Portuguese A/V Diarization Ground-Truth Pipeline

Builds speaker-turn-labeled transcripts from Easy Portuguese-style street
interview videos, using the burned-in bilingual captions as the transcript
source and video (face detection + a lightweight audio-visual sync signal)
to attribute each caption to a speaker.

Designed as an **annotation-assistance** pipeline, not an automatic ground
truth generator: every stage is built to make human verification fast
(pre-labeled candidates to confirm/correct), not to be trusted blind. See
"Known limitations" below for exactly where human review matters most.

## Pipeline stages

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `01_detect_shots.py` | Shot/cut boundaries | `data/raw/<id>.mp4` | `data/shots/<id>_shots.json` |
| `02_segment_captions.py` | Caption on/off segmentation | `data/raw/<id>.mp4` | `data/episodes/<id>_caption_segments.json` |
| `03_ocr_captions.py` | OCR the Portuguese caption line | segments + video | `data/episodes/<id>_transcript.json` |
| `04_track_faces.py` | Face detection + within-shot tracking | shots + video | `data/episodes/<id>_face_tracks.json` |
| `05_active_speaker.py` | Attribute captions to a track (heuristic, CPU) | tracks + transcript + audio | `data/episodes/<id>_speaker_attribution.json` |
| `05b_active_speaker_talknet.py` | Same, using real pretrained TalkNet-ASD (GPU) | tracks + transcript + video | `data/episodes/<id>_speaker_attribution_talknet.json` |

Run in order, passing the video id (filename without `.mp4`):

```bash
apt-get install -y tesseract-ocr tesseract-ocr-por   # OCR language pack
```

### Environment setup: use a separate venv per stage

The four stages pull in three unrelated ML frameworks (TensorFlow for MTCNN,
librosa's numba/llvmlite for the heuristic ASD, and PyTorch for TalkNet-ASD),
each with its own Python-version compatibility window. Forcing pip to
resolve all of them into one environment is a common source of install
failures, especially on a Python version newer than what one of those
frameworks currently ships wheels for. Use one venv per stage instead --
each is quick to create and there's no real downside to a bit of
duplication:

```bash
# Steps 1-3: shot detection, caption segmentation, OCR
#python3.11 -m venv venv_core
#source venv_core/bin/activate

conda create --name avprep_1 python=3.11
conda activate avprep_1

pip install -r scripts/requirements/requirements-core.txt
conda install -c conda-forge tesseract

python3 scripts/01_detect_shots.py portuguese_01
python3 scripts/02_segment_captions.py portuguese_01
python3 scripts/03_ocr_captions.py portuguese_01

# Step 4: face detection/tracking (TensorFlow via MTCNN)

pip install -r scripts/requirements/requirements-step4-face.txt
python3 scripts/04_track_faces.py portuguese_01
python3 scripts/04b_cluster_faces.py portuguese_01

# Create a file - data/corrections/portuguese_01_corrections.json, by looking at "data/episodes/portuguese_01_face_identities.json" and "outputs/portuguese_01_borderline_review.png". The png file will have the borderline confused speaker faces, or they can be some random image as well. The left column of the png file is what is confused, and the rigth column is what it is confused with. If the left and the rigfht column belong to the same person, identify the "shot" name from the png file and open the "data/episodes/portuguese_01_face_identities.json" file to get the exact person identity and include them in the "merge" section of "*corrections.json" file. If the left and right column belong to different persons, nothing needs to be done. And finally, if the images (left and right) do not belong to any humans at all, then add them under the "exclude" list in the "*corrections.json" file.

# Once the "*corrections.json" file is created, execute the following command.
python3 scripts/04c_apply_identity_corrections.py portuguese_01 data/corrections/portuguese_01_corrections.json

conda deactivate

# Step 5: heuristic ASD baseline (librosa)
#python3.11 -m venv venv_asd_heuristic
#source venv_asd_heuristic/bin/activate

conda create --name avprep_2 python=3.12
conda activate avprep_2

pip install -r scripts/requirements/requirements-step5-heuristic.txt

python3 scripts/05_active_speaker.py portuguese_01

# Step 5b: real TalkNet-ASD (PyTorch, needs a CUDA GPU)
python3.11 -m venv venv_talknet
source venv_talknet/bin/activate
# Install torch matching your CUDA version FIRST -- see
# https://pytorch.org/get-started/locally/ -- then the rest:
pip install torch torchvision torchaudio   # use the command from the selector above instead of this bare line
pip install -r scripts/requirements/requirements-step5b-talknet.txt

git clone https://github.com/TaoRuijie/TalkNet-ASD third_party/TalkNet-ASD
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python3 scripts/05b_active_speaker_talknet.py portuguese_01

# Execute as shown below if you want to make sure that the script doesn't take too much memory
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python3 scripts/05b_active_speaker_talknet.py portuguese_01 --max-gpu-mem-mb 1200

deactivate
```

**Recommended Python version: 3.10 or 3.11** for the `venv_face`,
`venv_asd_heuristic`, and `venv_talknet` environments -- TensorFlow, numba
(a librosa dependency), and PyTorch's wheel availability all tend to lag a
few months behind the newest Python release, so a brand-new Python (3.13+)
or an old one (<3.9) is the most likely reason a given package refuses to
install. `venv_core` is much less picky since it has no heavy ML
dependencies. Check what's available with `ls /usr/bin/python3.*` or
`pyenv versions`; if 3.10/3.11 isn't installed, `apt-get install
python3.11 python3.11-venv` (Ubuntu) or `pyenv install 3.11` gets you there
without disturbing the system Python.

If a specific package still fails to install after switching to 3.10/3.11,
the exact pip error (not just "incompatible") usually names the offending
package/version directly -- worth sharing that if it comes up.

Expects `data/raw/<id>.mp4` to exist; creates everything else.

## Step 5b details: real pretrained ASD

`05b_active_speaker_talknet.py` replaces the heuristic in step 5 with the
actual pretrained **TalkNet-ASD** model
(https://github.com/TaoRuijie/TalkNet-ASD), instead of the mouth-motion/audio
correlation proxy. It reuses this pipeline's own shot detection and MTCNN
face tracks (already validated) and only swaps in TalkNet-ASD for the
scoring step -- the crop/preprocessing math (crop scale, median-filter
smoothing, MFCC window/step, the 4:1 audio:video frame ratio the model's
cross-attention assumes) is ported faithfully from the official repo's own
`demoTalkNet.py`, not reimplemented from memory, specifically because subtle
transcription errors there would silently degrade a pretrained model's
output with no obvious error message. Setup is covered in the venv
walkthrough above.

The pretrained checkpoint (~50MB) auto-downloads from Google Drive via
`gdown` on first run. If Google's automated-download rate limit blocks
`gdown` (a known Google Drive quirk with heavily-shared files, not specific
to this repo), the script will tell you the file id so you can download it
manually and place it at `third_party/TalkNet-ASD/pretrain_TalkSet.model`.

**What's verified vs. not.** Everything up to the actual model forward pass
was tested against this pipeline's real footage in a sandboxed environment
without GPU access: the face-track-to-TalkNet-format conversion (11/11
tracks converted correctly), the 25fps re-encoding, the face-crop math
(visually confirmed correctly centered), and the audio/video alignment
(confirmed exactly 4:1, which is the ratio the model's cross-attention
depends on). **The only untested parts are the Google Drive weight download
and the actual CUDA inference itself** -- both require a real GPU + internet
access to check. If `loadParameters()` reports mismatched tensor shapes, or
inference throws a shape error, that's the first place to look, and worth
reporting back with the exact error.

**Output format** mirrors `05_active_speaker.py`'s so the two are directly
comparable: `assigned` (TalkNet scored the top candidate >= 0, its own
"speaking" threshold), `assigned_low_confidence` (a candidate was picked
but TalkNet's own score leaned toward "not speaking"), or
`uncertain_no_confident_face` (no track overlapped that segment at all).
Compare `speaker_attribution.json` (heuristic) against
`speaker_attribution_talknet.json` (real model) segment-by-segment to see
where they agree/disagree -- disagreements are the most useful places to
spot-check by eye.

## Calibration: this is tuned to ONE channel's template

The caption crop coordinates (`PT_LINE_FRAC` / `EN_LINE_FRAC` in scripts 02
and 03) and the text-shape thresholds (`WHITE_THRESH`, `MIN_TEXT_LIKE_COMPONENTS`,
etc. in script 02) were calibrated against Easy Portuguese's specific caption
template at 640x360. **Before running on a new channel or a noticeably
different resolution, recalibrate these against a sample frame** — see the
"Calibration walkthrough" section below for the exact process used.

## Why some things are heuristic, not state-of-the-art

This was built inside a sandboxed environment with a restricted network
allowlist (github.com, pypi.org, npm, apt mirrors -- **no Google Drive, no
git-lfs media, no arbitrary model hosting**). That ruled out the two
strongest options for both stages it affects:

- **Face detection (step 4):** stronger detectors (YuNet, RetinaFace)
  publish weights via git-lfs, which resolves through
  `media.githubusercontent.com` -- blocked here. Used **MTCNN** instead,
  since its weights ship inside the pip wheel itself. Validated against
  this footage: 4/4 spot-checked angled/profile faces detected at >0.95
  confidence, vs. 1/4 for a tuned Haar Cascade baseline (OpenCV's
  zero-dependency bundled option). **On a machine with normal internet
  access, swap in RetinaFace/YuNet/insightface for better accuracy and much
  better speed (GPU)** -- `04_track_faces.py` only needs a function that
  returns `(box, keypoints, confidence)` per face per frame, so the
  detector is a drop-in swap; the tracking logic doesn't change.

- **Active speaker detection (step 5):** the standard trained models
  (Light-ASD, TalkNet-ASD) host pretrained weights on Google Drive --
  unreachable here. Implemented the underlying signal-level idea instead
  (mouth-crop frame-difference correlated against audio RMS energy) as a
  baseline. **This is the weakest link in the pipeline** -- see limitations
  below. On an unrestricted machine, clone Light-ASD or TalkNet-ASD, pull
  weights via `gdown`, and replace `mouth_motion_signal()` with per-frame
  ASD network scores; everything downstream (segment-level aggregation,
  candidate ranking) still applies and gets simpler, since a trained model
  gives you a direct speaking probability instead of a proxy signal you
  then have to correlate.

## Known limitations (read before trusting output blind)

1. **ASD correlation is noisy at 5fps.** Validated on the test clip: it
   correctly picked the true speaker over a competing bystander face when
   two people were on screen together (the main case it needs to get
   right). But on clearly unambiguous single-speaker shots, the raw
   correlation is sometimes still weak or slightly negative just from
   signal noise -- e.g. one fully-visible, clearly-speaking subject scored
   -0.39 despite zero ambiguity, while a genuinely hard case (speaker
   nearly cropped out of frame) correctly scored -0.55. The decision logic
   in `05_active_speaker.py` accounts for this by treating correlation as
   a *disambiguator between multiple candidates*, not an absolute
   confidence gate -- a single visible candidate gets assigned by default
   with a `_low_conf` tag rather than being discarded, but a strong
   negative correlation (below `STRONG_ANTICORR_REJECT`) still blocks
   assignment, since that pattern also flagged a genuine off-screen-speaker
   case correctly. **A `_low_conf` tag means "trust the face, not the
   confidence number" -- worth a lighter human pass, not a rejection.**
2. **Off-screen speakers.** If no face is visible at all (pure b-roll, or
   camera on the wrong person during a question), there is no visual
   signal to attribute from. These segments come out as
   `uncertain_no_confident_face` and need either manual labeling or a
   voice-matching fallback (out of scope here).
3. **Cross-shot speaker identity is NOT tracked.** Every track is scoped
   to a single shot (`shot{N}_track{M}`); the same physical person
   appearing in two different shots gets two different IDs. Merging these
   into persistent per-video speaker identities needs face embedding
   similarity (e.g. ArcFace) across tracks -- not implemented.
4. **Real background text can pass the caption shape-gate.** A street
   sign or flyer with actual printed text, if it happens to sit in the
   fixed caption crop band and stays visually stable for >0.3s, would look
   like a caption to step 2's shape+stability heuristic. Didn't occur as a
   full false segment on the test clip, but isn't structurally ruled out --
   worth spot-checking segments against source frames on new videos.
5. **Single-video calibration.** Everything above was validated on one
   90-second clip from one channel. Before trusting this at scale, run it
   on a handful of videos spanning different hosts/lighting/backgrounds
   and re-check the failure modes above still hold.

## Calibration walkthrough (for a new channel/template)

1. Pull a handful of frames with `ffmpeg -i video.mp4 -vf fps=1 frame_%03d.png`
   and find one with the caption visible.
2. Crop candidate caption regions and inspect row-wise white-pixel density
   (see the row-brightness-profile approach used during development) to
   find the exact y-range of the Portuguese line vs. the English
   translation line -- look for a clean gap between two dense bands.
3. Update `PT_LINE_FRAC` / `EN_LINE_FRAC` in scripts 02 and 03.
4. Re-run step 2 and check the debug plot
   (`outputs/<id>_caption_signal.png`) for spurious spikes -- bright scene
   content (sky, pavement, water) is the usual false-positive source.
   Adjust `MIN_TEXT_LIKE_COMPONENTS` if needed.
5. Spot-check `outputs/<id>_ocr_review.png` against source frames for a
   sample of segments before trusting OCR output on a new channel/font.
