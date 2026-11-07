"""Step 3b: train and honestly evaluate a throw classifier on the CLIP frame embeddings.

    backend/.venv/Scripts/python training/03b_train_clip.py

Why two splits:
  * `official`: the dataset's own train / val / test folders. Clips cut from the SAME screen recording appear on both sides
    of that split, so it can reward recognising the session (people, room, camera) instead of the throw: optimistic.
  * `grouped` (the one to believe): every clip is grouped by its source recording and whole groups go to train, val or test.
Protocol: choose models and stop early on `val`; report on `test` only. Baselines (majority, logistic regression) are shown
beside the network. A 6-fold grouped cross-validation of the logistic-regression baseline gives a more stable estimate.
Saves backend/models/throw_clip.pt (trained on the grouped split's train set, early-stopped on its val set).
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.throw_clip import build_head  # noqa: E402  (the SAME model class the backend serves)

OUT_MODEL = ROOT.parent / "backend" / "models" / "throw_clip.pt"
OUT_MODEL.parent.mkdir(exist_ok=True)
SEED = 7
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

d = np.load(ROOT / "data" / "clip_feats.npz", allow_pickle=False)
X = d["feats"].astype(np.float32)  # (V, F, 512)
y, split, names = d["label"], d["split"].astype(str), d["name"].astype(str)
classes = [str(c) for c in d["classes"]]
V, F, D = X.shape
print(f"{V} clips, {F} frames, {D}-d embeddings, classes: {classes}")


def stem(n: str) -> str:
    s = re.sub(r"^cropped_", "", Path(n).stem)
    return re.sub(r"\s*\(\d+\)$", "", s)


groups = np.array([stem(n) for n in names])
print(f"{len(set(groups))} source recordings; {int((np.array([list(groups).count(g) for g in groups]) > 1).sum())} clips share a source with another clip")

# ---------------------------------------------------------------- splits
official = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
folds = [te for _, te in sgkf.split(X, y, groups)]
grouped = {"test": folds[0], "val": folds[1], "train": np.concatenate(folds[2:])}
for name, sp in (("official", official), ("grouped", grouped)):
    print(f"\n{name} split: " + ", ".join(f"{s}={len(ix)}" for s, ix in sp.items()))
    print("  train/test source overlap:", len(set(groups[sp['train']]) & set(groups[sp['test']])), "recordings")
    print("  test class counts:", dict(zip(classes, np.bincount(y[sp["test"]], minlength=len(classes)).tolist())))


def metrics(t, p):
    return {"accuracy": round(float(accuracy_score(t, p)), 4), "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 4)}


def pooled(A):  # mean over frames, re-normalised
    m = A.mean(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


results = {"classes": classes, "clips": int(V), "sources": int(len(set(groups)))}

# ---------------------------------------------------------------- baselines on both splits
def baselines(sp, tag):
    out = {}
    tr, va, te = sp["train"], sp["val"], sp["test"]
    out["majority"] = metrics(y[te], np.full(len(te), np.bincount(y[tr]).argmax()))
    for name, feat in (("logreg_mean", pooled), ("logreg_frames", lambda A: A.reshape(len(A), -1))):
        best = None
        for C in (0.3, 1, 3, 10, 30):
            m = LogisticRegression(C=C, max_iter=4000).fit(feat(X[tr]), y[tr])
            f1v = f1_score(y[va], m.predict(feat(X[va])), average="macro")
            if best is None or f1v > best[0]:
                best = (f1v, C, m)
        out[name] = {**metrics(y[te], best[2].predict(feat(X[te]))), "C": best[1], "val_macro_f1": round(float(best[0]), 4)}
    results[f"baselines_{tag}"] = out
    print(f"\n[{tag}] test-split results:")
    for k, v in out.items():
        print(f"   {k:14s} {v}")


baselines(official, "official")
baselines(grouped, "grouped")

# 6-fold grouped CV of the mean-embedding logistic regression: a steadier estimate than one 15 % test fold
cv = []
for k in range(6):
    te = folds[k]
    tr = np.concatenate([folds[j] for j in range(6) if j != k])
    m = LogisticRegression(C=3, max_iter=4000).fit(pooled(X[tr]), y[tr])
    cv.append(accuracy_score(y[te], m.predict(pooled(X[te]))))
results["cv6_grouped_logreg_mean_accuracy"] = {"mean": round(float(np.mean(cv)), 4), "std": round(float(np.std(cv)), 4), "folds": [round(float(c), 3) for c in cv]}
print(f"\n6-fold grouped CV, logistic regression on the mean embedding: {np.mean(cv):.3f} +/- {np.std(cv):.3f}  {np.round(cv, 3).tolist()}")


# ---------------------------------------------------------------- the temporal network (grouped split)
def augment(A):
    A = A + rng.normal(0, 0.02, A.shape).astype(np.float32)  # embedding noise
    shift = rng.integers(-1, 2, len(A))  # +-1 frame temporal jitter (edge frames repeat); the ORDER is never reversed
    idx = np.clip(np.arange(F)[None, :] + shift[:, None], 0, F - 1)
    A = np.take_along_axis(A, idx[:, :, None], 1)
    drop = rng.random((len(A), F, 1)) < 0.1  # drop a frame by copying its neighbour
    A = np.where(drop, np.roll(A, 1, axis=1), A)
    return A / np.linalg.norm(A, axis=2, keepdims=True)


tr, va, te = grouped["train"], grouped["val"], grouped["test"]
net = build_head(len(classes))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
counts = np.bincount(y[tr], minlength=len(classes))
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32), label_smoothing=0.1)
EPOCHS, PATIENCE = 150, 30


def probs_of(ix):
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(X[ix])), 1).numpy()


best, best_ep, bad = -1, 0, 0
t0 = time.time()
print("\n=== training the temporal head (grouped split) ===")
for ep in range(1, EPOCHS + 1):
    net.train()
    order = rng.permutation(tr)
    Xa = augment(X[order])
    tot = 0.0
    for b in range(0, len(order), 32):
        xb, yb = torch.from_numpy(Xa[b : b + 32]), torch.from_numpy(y[order[b : b + 32]])
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(yb)
    m = metrics(y[va], probs_of(va).argmax(1))
    if m["macro_f1"] > best:
        best, best_ep, bad = m["macro_f1"], ep, 0
        torch.save({"model_state_dict": net.state_dict(), "classes": classes, "n_frames": F, "dim": D, "hidden": 128,
                    "clip_model": "openai/clip-vit-base-patch16", "preprocess": "letterbox-square", "val": m, "best_epoch": ep}, OUT_MODEL)
    else:
        bad += 1
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:3d}  loss {tot / len(order):.3f}  val acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  (best {best:.3f} @ {best_ep})  {time.time() - t0:.0f}s")
    if bad >= PATIENCE:
        print("early stop")
        break

ck = torch.load(OUT_MODEL, map_location="cpu", weights_only=True)
net.load_state_dict(ck["model_state_dict"])
p = probs_of(te)
yhat = p.argmax(1)
results["network_grouped_test"] = {**metrics(y[te], yhat), "best_epoch": int(ck["best_epoch"]), "val_macro_f1": ck["val"]["macro_f1"]}
po = probs_of(official["test"])
results["network_official_test_(optimistic)"] = metrics(y[official["test"]], po.argmax(1))
conf = p.max(1)
results["confidence"] = {"mean_when_right": round(float(conf[yhat == y[te]].mean()), 3) if (yhat == y[te]).any() else None,
                         "mean_when_wrong": round(float(conf[yhat != y[te]].mean()), 3) if (yhat != y[te]).any() else None}
for thr in (0.5, 0.6, 0.7):
    keep = conf >= thr
    results[f"accuracy_when_confidence>={thr}"] = {"accuracy": round(float(accuracy_score(y[te][keep], yhat[keep])), 3) if keep.any() else None, "share_of_clips": round(float(keep.mean()), 3)}

print("\n=== FINAL ===")
for k in ("network_grouped_test", "network_official_test_(optimistic)", "confidence", "accuracy_when_confidence>=0.5", "accuracy_when_confidence>=0.6", "accuracy_when_confidence>=0.7"):
    print(f"  {k}: {results[k]}")
print("\n" + classification_report(y[te], yhat, labels=list(range(len(classes))), target_names=classes, zero_division=0))
cm = confusion_matrix(y[te], yhat, labels=list(range(len(classes))))
print("confusion matrix (rows = true, cols = predicted):\n", cm)
results["confusion_matrix"] = cm.tolist()
(ROOT / "results_clip.json").write_text(json.dumps(results, indent=2))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(confusion_matrix(y[te], yhat, labels=list(range(len(classes))), normalize="true"), display_labels=classes).plot(ax=ax, cmap="Blues", xticks_rotation=30, values_format=".2f")
    ax.set_title("Throw classifier (CLIP frames), grouped test split")
    fig.tight_layout()
    fig.savefig(ROOT / "confusion_matrix_clip.png", dpi=140)
except Exception as e:
    print("could not draw the confusion matrix:", e)
print(f"\nmodel: {OUT_MODEL}\nnumbers: {ROOT / 'results_clip.json'}")
