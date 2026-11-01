"""
Combine several trained ResNet-50 + VGGFace2 emotion models into one ensemble.

Steps
  1. Rebuild the SAME validation split used in training (split_seed 42, 10% of train).
  2. Get softmax probabilities (with horizontal-flip test-time augmentation) from every model
     on validation and test.
  3. Choose the best combination of models on VALIDATION macro-F1 (greedy forward selection).
  4. Optionally learn small per-class log-prior offsets on VALIDATION to balance precision/recall
     (e.g. fewer false "Disgust" predictions).
  5. Report TEST accuracy / macro-F1 once, and write models/emotion_ensemble.json, which app.py reads.

Usage (from D:\\Research4, with the venv):
  venv\\Scripts\\python.exe ensemble_emotion.py --data_dir TestDb/DATASET --models models/best_resnet50_vggface2_emotion.pth models/emotion_r50_s7_cw03.pth models/emotion_r50_s2024_cw03.pth
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from train_emotion_resnet50_vggface2 import (EMOTIONS, ImageList, VGGFace2ResNet50, build_transforms, list_images)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval(); out = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[3])), 1)
        out.append((p / 2).cpu())
    return torch.cat(out).numpy()


def score(probs, y, bias=None):
    logp = np.log(probs + 1e-9) + (0 if bias is None else bias)
    pred = logp.argmax(1)
    return f1_score(y, pred, average="macro"), (pred == y).mean(), pred


def fit_bias(probs, y, steps=(-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4), rounds=2):
    """Coordinate search of one additive log-prior offset per class, maximising validation macro-F1."""
    bias = np.zeros(probs.shape[1])
    best = score(probs, y, bias)[0]
    for _ in range(rounds):
        for c in range(probs.shape[1]):
            for d in steps:
                b = bias.copy(); b[c] = d
                f = score(probs, y, b)[0]
                if f > best + 1e-6:
                    best, bias = f, b
    return bias, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_bias", action="store_true", help="skip per-class offset calibration")
    ap.add_argument("--out_json", default="models/emotion_ensemble.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    models = [m for m in a.models if os.path.exists(m)]
    for m in a.models:
        if m not in models:
            print("[skip] not found:", m)
    if not models:
        raise SystemExit("No model files found.")

    train_all = list_images(os.path.join(a.data_dir, "train"))
    labels = np.array([y for _, y in train_all])
    _, va_idx = next(StratifiedShuffleSplit(1, test_size=a.val_frac, random_state=a.split_seed).split(labels, labels))
    _, test_tf = build_transforms(a.img_size)
    va = ImageList([train_all[i] for i in va_idx], test_tf)
    te = ImageList(list_images(os.path.join(a.data_dir, "test")), test_tf)
    vl = DataLoader(va, a.batch_size, num_workers=a.workers)
    tl = DataLoader(te, a.batch_size, num_workers=a.workers)
    yv = np.array([y for _, y in va.items]); yt = np.array([y for _, y in te.items])

    val_p, test_p = {}, {}
    for m in models:
        ck = torch.load(m, map_location=device)
        if ck.get("split_seed", 42) != a.split_seed or abs(ck.get("val_frac", a.val_frac) - a.val_frac) > 1e-9:
            print(f"[warn] {m} was trained with a different validation split - its validation score is not fair")
        net = VGGFace2ResNet50(num_classes=7); net.load_state_dict(ck["state_dict"]); net.to(device)
        val_p[m], test_p[m] = predict_probs(net, vl, device), predict_probs(net, tl, device)
        f, acc, _ = score(val_p[m], yv)
        print(f"[model] {os.path.basename(m)}: val acc {acc*100:.2f}% | val macro-F1 {f*100:.2f}%")
        del net; torch.cuda.empty_cache()

    # choose the subset with best validation macro-F1 (all subsets; at most a handful of models)
    best = None
    for r in range(1, len(models) + 1):
        for subset in itertools.combinations(models, r):
            f, acc, _ = score(np.mean([val_p[m] for m in subset], 0), yv)
            if best is None or f > best[0] + 1e-6:
                best = (f, acc, subset)
    f, acc, chosen = best
    print(f"\n[ensemble] chosen {len(chosen)} model(s) on validation: {[os.path.basename(m) for m in chosen]}")
    print(f"[ensemble] val acc {acc*100:.2f}% | val macro-F1 {f*100:.2f}%")

    pv = np.mean([val_p[m] for m in chosen], 0); pt = np.mean([test_p[m] for m in chosen], 0)
    bias = np.zeros(7)
    if not a.no_bias:
        bias, fb = fit_bias(pv, yv)
        print(f"[bias] per-class offsets {np.round(bias, 2).tolist()} -> val macro-F1 {fb*100:.2f}%")
        if fb - f < 0.003:  # ignore tiny gains: likely noise on 1,228 images
            bias = np.zeros(7); print("[bias] gain too small, offsets not used")

    tf1, tacc, pred = score(pt, yt, bias)
    print(f"\n[TEST] accuracy {tacc*100:.2f}% | macro-F1 {tf1*100:.2f}%")
    print(classification_report(yt, pred, target_names=EMOTIONS, digits=4, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(yt, pred))
    from fairness import disparity_summary, per_class_metrics
    ds = disparity_summary(per_class_metrics(yt, pred, EMOTIONS), yt, pred)
    print(f"[fairness] balanced accuracy {ds['balanced_accuracy']*100:.2f}% | worst class {ds['worst_recall_class']} "
          f"recall {ds['worst_recall']*100:.1f}% | recall ratio {ds['recall_ratio']:.2f} "
          f"({'passes' if ds['passes_four_fifths'] else 'below'} 0.80) - full audit: python fairness.py --data_dir {a.data_dir}")

    with open(a.out_json, "w") as fjs:
        json.dump({"models": ["models/" + os.path.basename(m) for m in chosen],  # app.py looks in its models/ folder
 "class_bias": bias.round(4).tolist(),
                   "classes": EMOTIONS, "img_size": a.img_size, "val_macro_f1": round(float(f), 4),
                   "test_accuracy": round(float(tacc), 4), "test_macro_f1": round(float(tf1), 4)}, fjs, indent=2)
    print("[saved]", a.out_json, "- restart the app to use it")


if __name__ == "__main__":
    main()
