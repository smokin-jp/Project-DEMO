# Judo throw model (training)

Turns the Hugging Face dataset [`adenhaus/judo_throws`](https://huggingface.co/datasets/adenhaus/judo_throws) (801 short clips, 4 throws: **ippon seoi nage, o goshi, osoto gari, uchi mata**, MIT licence) into the model the backend serves at `/api/throw_type`.

## Reproduce

```bash
python -m venv training/.venv
training/.venv/Scripts/pip install huggingface_hub opencv-python-headless                                  # dataset download
backend/.venv/Scripts/pip install opencv-python-headless scikit-learn matplotlib                         # CLIP features + training

training/.venv/Scripts/python training/01_download.py          # ~470 MB of videos into training/data/raw (skips the 1.8 GB full.zip)
backend/.venv/Scripts/python training/02b_extract_clip_features.py   # 8 CLIP embeddings per clip, ~25 min on a laptop CPU, resumable
backend/.venv/Scripts/python training/03b_train_clip.py        # train + honest evaluation -> backend/models/throw_clip.pt, results_clip.json
backend/.venv/Scripts/python training/03c_cv_clip.py           # 6-fold grouped cross-validation (steadier number)
```

## What was tried, and what happened

1. **Skeleton route: tried first and abandoned (its scripts were removed).** The app's own MediaPipe pose model finds **both fighters in only 2 % of frames** with continuous tracking (15 % with isolated frames); in a throw the bodies overlap and it mostly sees one person. A model that needs two clean skeletons cannot work on this footage. (This is also a warning for the live app: the leg-reap throw rule needs both fighters visible.)
2. **CLIP route (`02b`, `03b`, `03c`): used.** 8 evenly spaced frames per clip (letterboxed to a square, not cropped) -> the backend's CLIP image encoder -> a ~100 k-parameter temporal head (per-frame projection, small temporal convolution, mean+max pooling). Training and serving import the same classes from `backend/app/throw_clip.py`, so they cannot drift apart.

## Results (read these, not just the biggest number)

| | accuracy |
|---|---|
| Always guess the commonest class | 22-28 % |
| Official train / val / test folders, logistic regression | 88-96 % (**inflated: see below**) |
| Official test folder, the network | 95 % (**inflated**) |
| **Grouped split (honest), the network** | **85 %** (macro-F1 0.85) on 133 held-out clips |
| **6-fold grouped cross-validation, the network** | **83.5 % +/- 3.6 %** (folds 76-87 %) |
| Through the live `/api/throw_type` endpoint, 32 held-out clips | 78 % (about +/-7 % with 32 clips) |

- **Why the official split is inflated:** 125 of 677 source recordings (the clips are cuts of screen recordings, plus `cropped_` / `ezgif` re-cuts) appear in more than one of the dataset's splits, and 60 recordings sit on both sides of train/test. Even a plain logistic regression reaches 96 % there, which means recognising the session (people, room, camera), not the throw. The grouped split keeps every clip of a source recording on one side.
- **Confidence is informative:** mean top-class probability 0.85 when right and 0.59 when wrong; accuracy rises to 89 % / 93 % / 95 % when only answers with confidence >= 0.5 / 0.6 / 0.7 are kept (keeping 93 % / 85 % / 77 % of clips). The app scores a throw only at confidence >= 0.5.
- Worst confusions: ippon seoi nage vs osoto gari / uchi mata; o goshi is the cleanest (precision 1.00).

## Limits (honest)

- **Domain shift.** Every clip is a screen recording of the same kind of footage. A live camera in a gym (angle, lighting, clothing, speed) will look different; expect lower accuracy than the numbers above. Nothing here has been run on live video.
- **Only 4 throws, and no "not a throw" class.** The model always names one of the four. The app therefore only asks it after a firm fall next to the opponent (`src/throwgrade.js: decideThrow`), and scores only a confident answer.
- Small data (801 clips, 676 sources): the cross-validation spread of +/-3.6 % is real.
- The labels were taken from the folder names; the dataset README is empty, so how the clips were made and checked is unknown.
- The 4 `.DS_Store.mp4` files in the repo are macOS junk and are skipped.
