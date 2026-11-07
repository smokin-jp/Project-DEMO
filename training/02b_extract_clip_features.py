"""Step 2b: CLIP image embeddings for every clip (8 frames each), using the SAME CLIP model the backend already runs.

Pose cannot do this job on this footage: during a throw the bodies overlap and MediaPipe finds both fighters in only ~2-15 %
of frames. CLIP looks at pixels, so occlusion does not break it.

    backend/.venv/Scripts/python training/02b_extract_clip_features.py

Resumable: results are saved every 40 clips; run it again to continue. Output: training/data/clip_feats.npz
    feats (V, F, 512) L2-normalised CLIP image embeddings, label (V,), split (V,), name (V,), classes (4,)
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.clip_model import ClipScorer  # noqa: E402
from app.config import settings  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clip_feats.npz"
F = 8  # frames per clip, evenly spaced over the whole clip

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))


def letterbox(img: Image.Image) -> Image.Image:
    """pad to a square instead of centre-cropping, so a fighter at the side of a wide frame is not cut off"""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def frames_of(path: Path) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for fi in np.linspace(0, max(n - 1, 0), F).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            ok_prev = out[-1] if out else None
            if ok_prev is not None:
                out.append(ok_prev)  # repeat the last good frame
            continue
        out.append(letterbox(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    while out and len(out) < F:
        out.append(out[-1])
    return out


def main():
    videos = sorted(v for v in RAW.glob("*/*/*.mp4") if not v.name.startswith("."))
    classes = sorted({v.parts[-2] for v in videos})
    key = lambda v: f"{v.parts[-3]}/{v.parts[-2]}/{v.name}"  # noqa: E731
    done = {}
    if OUT.exists():
        d = np.load(OUT, allow_pickle=False)
        for k, f in zip(d["key"], d["feats"]):
            done[str(k)] = f
        print(f"resuming: {len(done)} clips already done")
    todo = [v for v in videos if key(v) not in done]
    print(f"{len(videos)} clips, {len(todo)} to do, {F} frames each", flush=True)

    scorer = ClipScorer(settings.clip_model, "cpu").load()
    t0 = time.time()

    def save():
        ks = [key(v) for v in videos if key(v) in done]
        vs = [v for v in videos if key(v) in done]
        np.savez_compressed(
            OUT,
            feats=np.stack([done[k] for k in ks]).astype(np.float32),
            key=np.array(ks),
            label=np.array([classes.index(v.parts[-2]) for v in vs], np.int64),
            split=np.array([v.parts[-3] for v in vs]),
            name=np.array([v.name for v in vs]),
            classes=np.array(classes),
        )

    for i, v in enumerate(todo, 1):
        imgs = frames_of(v)
        if not imgs:
            print("  unreadable, skipped:", v.name)
            continue
        done[key(v)] = scorer.embed(imgs).numpy()[:F]
        if i % 40 == 0 or i == len(todo):
            save()
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el:5.0f}s  (~{el / i * (len(todo) - i):.0f}s left)", flush=True)
    save()
    print(f"saved {OUT} ({len(done)} clips, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
