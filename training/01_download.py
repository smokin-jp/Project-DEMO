"""Step 1: download the judo-throw videos (train/val/test folders only; skips the 1.8 GB full.zip duplicate) and inspect them.

    training/.venv/Scripts/python training/01_download.py
"""
from collections import Counter, defaultdict
from pathlib import Path
import re

import cv2
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"

if not any(RAW.glob("train/*/*.mp4")):
    snapshot_download(
        repo_id="adenhaus/judo_throws",
        repo_type="dataset",
        allow_patterns=["train/**", "val/**", "test/**"],
        ignore_patterns=["**/.DS_Store"],
        local_dir=str(RAW),
    )

videos = sorted(v for v in RAW.glob("*/*/*.mp4") if not v.name.startswith("."))  # skip macOS .DS_Store.mp4 junk
print(f"\n{len(videos)} videos in {RAW}")

by_split = defaultdict(Counter)
meta = []
for v in videos:
    split, cls = v.parts[-3], v.parts[-2]
    by_split[split][cls] += 1
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    meta.append((v, split, cls, fps, n, w, h))

classes = sorted({m[2] for m in meta})
print("classes:", classes)
print(f"\n{'split':6s} " + " ".join(f"{c:>16s}" for c in classes) + "   total")
for split in ("train", "val", "test"):
    print(f"{split:6s} " + " ".join(f"{by_split[split][c]:16d}" for c in classes) + f"   {sum(by_split[split].values())}")

durs = [m[4] / m[3] for m in meta if m[3] > 0]
print(f"\nduration s: min {min(durs):.2f}  median {sorted(durs)[len(durs)//2]:.2f}  max {max(durs):.2f}")
print("fps:", dict(Counter(round(m[3]) for m in meta)))
print("resolutions (top 5):", Counter((m[5], m[6]) for m in meta).most_common(5))
bad = [m[0].name for m in meta if m[4] <= 0]
print("unreadable / empty videos:", len(bad), bad[:5])

# Possible leakage: the same source recording appears in several splits (cropped_/ezgif variants of one screen recording)
def stem(name: str) -> str:
    s = re.sub(r"^cropped_", "", Path(name).stem)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return s
groups = defaultdict(set)
for v, split, cls, *_ in meta:
    groups[stem(v.name)].add(split)
shared = {k: sorted(s) for k, s in groups.items() if len(s) > 1}
print(f"\nsource names appearing in more than one split (possible train/test leakage): {len(shared)} of {len(groups)}")
for k, s in list(shared.items())[:8]:
    print("  ", k, s)
