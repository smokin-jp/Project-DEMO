"""Step 3c: 6-fold grouped cross-validation of the temporal head (a steadier accuracy estimate than one test fold).

    backend/.venv/Scripts/python training/03c_cv_clip.py

For fold k: test = fold k, validation (early stopping / epoch choice) = fold k+1, train = the other four. Clips from one source
recording are never split across folds. Same model, augmentation and recipe as 03b_train_clip.py.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.throw_clip import build_head  # noqa: E402

SEED = 7
d = np.load(ROOT / "data" / "clip_feats.npz", allow_pickle=False)
X, y, names = d["feats"].astype(np.float32), d["label"], d["name"].astype(str)
classes = [str(c) for c in d["classes"]]
V, F, D = X.shape
groups = np.array([re.sub(r"\s*\(\d+\)$", "", re.sub(r"^cropped_", "", Path(n).stem)) for n in names])
folds = [te for _, te in StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED).split(X, y, groups)]


def run(k):
    rng = np.random.default_rng(SEED + k)
    torch.manual_seed(SEED + k)
    te, va = folds[k], folds[(k + 1) % 6]
    tr = np.concatenate([folds[j] for j in range(6) if j not in (k, (k + 1) % 6)])
    net = build_head(len(classes))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
    counts = np.bincount(y[tr], minlength=len(classes))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32), label_smoothing=0.1)

    def augment(A):
        A = A + rng.normal(0, 0.02, A.shape).astype(np.float32)
        idx = np.clip(np.arange(F)[None, :] + rng.integers(-1, 2, len(A))[:, None], 0, F - 1)
        A = np.take_along_axis(A, idx[:, :, None], 1)
        A = np.where(rng.random((len(A), F, 1)) < 0.1, np.roll(A, 1, axis=1), A)
        return A / np.linalg.norm(A, axis=2, keepdims=True)

    def pred(ix):
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(torch.from_numpy(X[ix])), 1).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(150):
        net.train()
        order = rng.permutation(tr)
        Xa = augment(X[order])
        for b in range(0, len(order), 32):
            opt.zero_grad()
            loss_fn(net(torch.from_numpy(Xa[b : b + 32])), torch.from_numpy(y[order[b : b + 32]])).backward()
            opt.step()
        f1 = f1_score(y[va], pred(va).argmax(1), average="macro")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k_: v.clone() for k_, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= 30:
            break
    net.load_state_dict(best_state)
    p = pred(te)
    return float(accuracy_score(y[te], p.argmax(1))), float(f1_score(y[te], p.argmax(1), average="macro")), len(te)


accs, f1s = [], []
for k in range(6):
    a, f, n = run(k)
    accs.append(a)
    f1s.append(f)
    print(f"fold {k}: n={n:3d}  accuracy {a:.3f}  macro-F1 {f:.3f}", flush=True)
print(f"\n6-fold grouped CV, temporal network: accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f}   macro-F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
res = json.loads((ROOT / "results_clip.json").read_text())
res["cv6_grouped_network"] = {"accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_std": round(float(np.std(accs)), 4), "macro_f1_mean": round(float(np.mean(f1s)), 4), "folds": [round(a, 3) for a in accs]}
(ROOT / "results_clip.json").write_text(json.dumps(res, indent=2))
