# FIX11 — ResNet-50 + VGGFace2 | Train on RAF-AU, Test on RAF-DB

**Scope:** ResNet-50 backbone with VGGFace2 face pretraining, **trained on RAF-AU** and **tested on RAF-DB** (cross-dataset). Weighted-CE baseline vs. balanced-sampling mitigation, per-class fairness audit, Grad-CAM.

The RAF-AU action-unit / InceptionResnetV1 / FACS-rules code is **out of scope**. Move `train_rafau.py` and the old `app.py` into `archive/`.

---

## 0. What the data actually contains (checked on your `D:\Research4`)

| Finding | Evidence | Consequence |
|---|---|---|
| **RAF-AU has no emotion labels.** `RAFAU_label.txt` = image name + AU codes only (4,601 lines, 77 AU tokens, 253 `null`) | Every line is `XXXX.jpg <AUs>`; the extra "third field" on 421 lines is just trailing whitespace | A 7-class emotion model cannot be trained from RAF-AU labels alone |
| **RAF-AU images are the RAF-ML images.** Same file names, pixel-identical (similarity 1.000 on all sampled pairs); all 4,601 RAF-AU names exist in RAF-ML | `RAF-ML/Image/aligned.zip` vs `RAF-AU/aligned/` | Emotion labels for RAF-AU training come from **`RAF-ML/EmoLabel/distribution.txt`** (crowd-sourced, 6 emotions) |
| **RAF-ML labels have 6 emotions, no Neutral**: Surprise, Fear, Disgust, Happiness, Sadness, Anger | RAF-ML README | Same order as RAF-DB folders 1–6. **RAF-DB folder 7 (Neutral, 680 test images) has no training class** |
| RAF-AU `null` (no AU) ≠ Neutral | The 253 `null` images have ordinary emotion distributions (mostly Disgust, Sadness, Anger) | Neutral can't be recovered from AU labels either |
| Labels are ambiguous (blended emotions) | Median max-probability is 0.56; only 1,709 of 4,601 images reach ≥ 0.6 | Train on the **soft distribution**, not just the argmax (default `--labels soft`) |
| **Train/test leakage exists**: RAF-DB test images also appear in RAF-AU | 1 of 50 sampled RAF-DB test images is the same photo as RAF-AU `0518` (similarity 0.955; the closest non-matches reach 0.907). Projected ≈ 2% of the test set | The training script now **removes duplicates** before training and writes a CSV for manual review |
| Label shift | RAF-AU argmax: Disgust 1073, Surprise 1008, Anger 903, Sadness 749, Happiness 515, Fear 353. RAF-DB test (1–6): Happiness 1185, Sadness 478, Surprise 329, Anger 162, Disgust 160, Fear 74 | Report **Macro F1** as the primary metric; accuracy alone will mislead |

**Decision taken: 6-class evaluation.** The model is scored on RAF-DB folders 1–6 (2,388 images). For the 680 Neutral images, the report lists what the model predicts but does not score them. The only way to score Neutral is to train on RAF-DB-train Neutral images, which breaks the "RAF-AU train / RAF-DB test" protocol.

**Don't compare this directly with 83.67%.** That number is in-domain (RAF-DB train → RAF-DB test, 7 classes). This setup is cross-dataset with 6 classes and 4.6k training images instead of 12k, so expect lower numbers. Present it as a **generalisation / cross-dataset** experiment.

All code below was smoke-tested on your real label files and a sample of your real images: dataset join, duplicate removal (it caught the real RAF-DB↔RAF-AU duplicate and a planted mirrored greyscale copy), both imbalance modes, soft and hard labels, test eval, Grad-CAM, `/predict`, `/generate-report`. The model weights in the test were fake, so the accuracy numbers mean nothing.

---

## Fix summary

| # | Problem | Fix |
|---|---|---|
| 1 | Label 7 = **Neutral**, not Contempt (notebook, old app, PPT) | Corrected; Neutral is reported, not scored |
| 2 | RAF-AU has no emotion labels | Joined with RAF-ML `distribution.txt` by file name |
| 3 | RAF-AU ↔ RAF-DB test duplicates | Removed automatically (`--dup_threshold 0.93`, mirror-aware); `leakage_rafau_vs_rafdb_test.csv` |
| 4 | ResNet-50 used ImageNet weights | Official VGGFace2 ResNet-50 in an exact Caffe-style ResNet-50 (loading into `torchvision.resnet50` gives no error but wrong features) |
| 5 | Model selected on the test set | Stratified 15% validation split of RAF-AU; RAF-DB test used once |
| 6 | Weighted CE **and** sampler together | `--imbalance weighted_ce` (baseline) or `sampler` (mitigation) |
| 7 | 1 epoch, no scheduler or early stopping | 30 epochs, ReduceLROnPlateau on val Macro F1, patience 7, 10× head LR, AMP |
| 8 | No eval, fairness or Grad-CAM for the new model | `results_*.json` + Grad-CAM folders |
| 9 | App silently ran with random weights; `D:\` paths; 35-image unseeded report; blocking `async` | Fails loudly; env-var paths; full test set; seeded; sync `def` |
| 10 | Notebook: `total_mem`, `verbose=True`, Grad-CAM hooks pile up | §6 |

---

## 1. Get the VGGFace2 ResNet-50 weights

1. Go to **https://github.com/cydonia999/VGGFace2-pytorch**. Its README links the converted Oxford VGG weights.
2. Download **`resnet50_ft_weight.pkl`** (recommended) or `resnet50_scratch_weight.pkl`.
3. Save it as `weights\resnet50_ft_weight.pkl`.

The loader requires **265/265** backbone tensors to load and raises an error otherwise.

Project layout:
```
D:\Research4\
├── vggface2_resnet.py        # NEW
├── train_resnet50.py         # REPLACED
├── app.py                    # REPLACED
├── static\index.html         # small JS change (§5)
├── weights\resnet50_ft_weight.pkl
├── models\                   # checkpoints, results_*.json, gradcam_*\, leakage csv
├── RAF-AU-20261003T122554Z-1-001\RAF-AU\{RAFAU_label.txt, aligned\}
├── RAF-ML-20261003T121432Z-1-001\RAF-ML\EmoLabel\distribution.txt
├── TestDb\DATASET\test\{1..7}\
└── archive\  train_rafau.py, old app.py
```
Install: `pip install torch torchvision grad-cam fpdf2 fastapi uvicorn python-multipart scikit-learn pillow numpy`

---

## 2. `vggface2_resnet.py` (NEW)

```python
"""
VGGFace2-pretrained ResNet-50: trained on RAF-AU (emotion labels from RAF-ML), tested on RAF-DB.

Weights: Oxford VGG official VGGFace2 ResNet-50 (Caffe), PyTorch port by
cydonia999/VGGFace2-pytorch -> file `resnet50_ft_weight.pkl`
(ft = MS1M pretrain + VGGFace2 fine-tune; `resnet50_scratch_weight.pkl` = VGGFace2 only).

IMPORTANT differences from torchvision.resnet50 (why we do NOT just load into torchvision):
  * Caffe-style bottleneck: stride sits on the first 1x1 conv, not the 3x3 conv.
  * maxpool uses padding=0, ceil_mode=True.
  * Input is BGR, 0-255 range, mean-subtracted (no std division).
Loading these weights into torchvision's resnet50 "works" (same key names/shapes) but
silently gives wrong features. This file mirrors the original architecture exactly.
"""
import os
import pickle

import torch
import torch.nn as nn

# ── Emotion classes ──
# TRAIN: RAF-AU images (identical to RAF-ML images, same file names). RAF-AU has only AU labels,
#        so emotion labels come from RAF-ML EmoLabel/distribution.txt, whose 6 columns are
#        Surprise, Fear, Disgust, Happiness, Sadness, Anger. No Neutral.
# TEST:  RAF-DB basic (folders 1..7). Folders 1..6 use the same order as RAF-ML -> index 0..5.
#        Folder 7 = Neutral (NOT Contempt) has no training data, so it is reported separately
#        and not scored.
EMOTION_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger"]
NUM_CLASSES = 6
IDX_TO_EMOTION = dict(enumerate(EMOTION_NAMES))
RAFDB_FOLDER_TO_IDX = {str(i + 1): i for i in range(6)}   # "1".."6" -> 0..5
RAFDB_NEUTRAL_FOLDER = "7"
RAFDB_LABELS = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness", 5: "Sadness", 6: "Anger", 7: "Neutral"}

# VGGFace2 preprocessing constants (BGR order, 0-255 scale)
VGGFACE2_MEAN_BGR = (91.4953, 103.8827, 131.0912)
IMG_SIZE = 224


class VGGFace2Normalize:
    """Tensor (RGB, 0-1) -> (BGR, 0-255, mean-subtracted). Picklable (safe for num_workers>0 on Windows)."""
    def __init__(self, mean_bgr=VGGFACE2_MEAN_BGR):
        self.mean = torch.tensor(mean_bgr).view(3, 1, 1)

    def __call__(self, x):
        x = x[[2, 1, 0], :, :] * 255.0
        return x - self.mean

    def denormalize(self, x):
        """Inverse, for visualisation: returns RGB 0-1."""
        x = (x + self.mean.to(x.device)) / 255.0
        return x[[2, 1, 0], :, :].clamp(0, 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class VGGFace2ResNet50(nn.Module):
    def __init__(self, num_classes=8631):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def load_vggface2_weights(model, pkl_path):
    """Load cydonia999 .pkl weights. Raises if the file is missing or most keys fail to load
    (no silent fallback to random/ImageNet weights)."""
    if not pkl_path or not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"VGGFace2 weights not found: {pkl_path!r}. Download resnet50_ft_weight.pkl "
            "(see FIX11.md, step 1) — refusing to train without them.")
    with open(pkl_path, "rb") as f:
        weights = pickle.load(f, encoding="latin1")
    own = model.state_dict()
    loaded, skipped = 0, []
    for name, arr in weights.items():
        if name.startswith("fc."):
            continue  # 8631-identity head is discarded
        if name in own and tuple(own[name].shape) == tuple(arr.shape):
            own[name].copy_(torch.from_numpy(arr))
            loaded += 1
        else:
            skipped.append(name)
    expected = sum(1 for k in own if not k.startswith("fc.") and not k.endswith("num_batches_tracked"))
    if loaded < expected:
        raise RuntimeError(f"Only {loaded}/{expected} VGGFace2 tensors loaded. Skipped: {skipped[:10]}")
    print(f"[OK] VGGFace2 weights loaded: {loaded}/{expected} tensors from {pkl_path}")
    return model


def build_model(pkl_path=None, num_classes=NUM_CLASSES, pretrained=True):
    model = VGGFace2ResNet50(num_classes=8631)
    if pretrained:
        load_vggface2_weights(model, pkl_path)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

---

## 3. `train_resnet50.py` (REPLACE)

```python
"""
ResNet-50 + VGGFace2 emotion recognition.
  TRAIN : RAF-AU aligned images (4,601). Emotion labels = RAF-ML distribution.txt (same images).
          6 classes: Surprise, Fear, Disgust, Happiness, Sadness, Anger.
  VAL   : stratified 15% of RAF-AU (model selection, LR schedule, early stopping).
  TEST  : RAF-DB basic test set, folders 1-6 (cross-dataset). Folder 7 (Neutral) has no
          training class: its prediction spread is reported, but it is not scored.
  LEAK  : RAF-AU images that duplicate a RAF-DB test image are removed before training.

Baseline vs mitigation:
  python train_resnet50.py --rafau_dir ... --rafml_dir ... --rafdb_test_dir ... --weights ... --imbalance weighted_ce
  python train_resnet50.py ... --imbalance sampler
"""
import os
import csv
import json
import time
import random
import argparse
import warnings
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

from vggface2_resnet import (build_model, VGGFace2Normalize, EMOTION_NAMES, IDX_TO_EMOTION,
                             NUM_CLASSES, IMG_SIZE, VGGFACE2_MEAN_BGR,
                             RAFDB_FOLDER_TO_IDX, RAFDB_NEUTRAL_FOLDER)

warnings.filterwarnings("ignore")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ───────────────────────── Data ─────────────────────────

def load_rafau_with_emotions(rafau_dir, rafml_dir):
    """Join RAF-AU image list (+AU strings) with RAF-ML 6-dim emotion distributions by file name."""
    dist = {}
    with open(os.path.join(rafml_dir, "EmoLabel", "distribution.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 7:
                dist[p[0]] = np.array(p[1:7], dtype=np.float32)
    items, missing_img, missing_lbl = [], 0, 0
    with open(os.path.join(rafau_dir, "RAFAU_label.txt")) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            name = p[0]
            img = os.path.join(rafau_dir, "aligned", f"{os.path.splitext(name)[0]}_aligned.jpg")
            if not os.path.isfile(img):
                missing_img += 1
                continue
            if name not in dist:
                missing_lbl += 1
                continue
            d = dist[name] / dist[name].sum()
            items.append({"name": name, "path": img, "soft": d, "hard": int(d.argmax()),
                          "aus": p[1] if len(p) > 1 else "null"})
    print(f"RAF-AU: {len(items)} images with emotion labels "
          f"(missing image: {missing_img}, missing RAF-ML label: {missing_lbl})")
    if not items:
        raise RuntimeError("No RAF-AU images matched RAF-ML labels — check --rafau_dir / --rafml_dir")
    return items


def load_rafdb_test(test_dir):
    scored, neutral = [], []
    for folder in sorted(os.listdir(test_dir)):
        path = os.path.join(test_dir, folder)
        if not os.path.isdir(path):
            continue
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if os.path.splitext(f)[1].lower() in VALID_EXT]
        if folder in RAFDB_FOLDER_TO_IDX:
            scored += [(p, RAFDB_FOLDER_TO_IDX[folder]) for p in files]
        elif folder == RAFDB_NEUTRAL_FOLDER:
            neutral += files
    if len({y for _, y in scored}) != NUM_CLASSES:
        raise RuntimeError(f"Expected RAF-DB folders 1-6 in {test_dir}")
    print(f"RAF-DB test: {len(scored)} scored images (folders 1-6), {len(neutral)} Neutral (folder 7, not scored)")
    return scored, neutral


def _fingerprint(path):
    """32x32 grayscale, zero-mean, unit-norm: robust to colour/compression differences."""
    im = Image.open(path).convert("L").resize((32, 32))
    out = []
    for img in (im, im.transpose(Image.FLIP_LEFT_RIGHT)):
        v = np.asarray(img, dtype=np.float32).ravel()
        v -= v.mean()
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return out


def remove_test_overlap(train_items, test_paths, out_csv, threshold=0.93):
    """Drop training images that are near-duplicates of any RAF-DB test image (incl. mirrored)."""
    tr = np.stack([_fingerprint(it["path"])[0] for it in train_items])
    te = [_fingerprint(p) for p in test_paths]
    te_n = np.stack([a for a, _ in te])
    te_f = np.stack([b for _, b in te])
    sim = np.maximum(tr @ te_n.T, tr @ te_f.T)          # (n_train, n_test)
    best_j = sim.argmax(1)
    best_s = sim[np.arange(len(train_items)), best_j]
    dup = best_s >= threshold
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rafau_image", "rafdb_test_image", "similarity", "removed"])
        for i in np.where(best_s >= threshold - 0.03)[0]:   # also log borderline cases for manual review
            w.writerow([train_items[i]["name"], os.path.basename(test_paths[best_j[i]]),
                        f"{best_s[i]:.4f}", bool(dup[i])])
    kept = [it for it, d in zip(train_items, dup) if not d]
    print(f"Leakage check: removed {int(dup.sum())} RAF-AU images duplicating RAF-DB test "
          f"(threshold {threshold}); report -> {out_csv}")
    return kept, int(dup.sum())


class ImageDataset(Dataset):
    """items: list of (path, soft_label_vector or None, hard_label int)."""
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, soft, hard = self.items[i]
        x = self.transform(Image.open(path).convert("RGB"))
        if soft is None:
            soft = np.eye(NUM_CLASSES, dtype=np.float32)[hard]
        return x, torch.from_numpy(soft), hard


def build_transforms():
    norm = VGGFace2Normalize()
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])
    return train_tf, eval_tf


# ───────────────────────── Loss / loops ─────────────────────────

class SoftTargetCE(nn.Module):
    """Cross-entropy against RAF-ML's crowd-sourced emotion distribution (optionally class-weighted).
    With one-hot targets this equals standard (weighted) cross-entropy."""
    def __init__(self, class_weights=None):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None else torch.ones(NUM_CLASSES))

    def forward(self, logits, soft):
        logp = F.log_softmax(logits.float(), dim=1)
        per_sample = -(soft * self.w * logp).sum(1)
        norm = (soft * self.w).sum(1)
        return (per_sample / norm).mean()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, soft, hard in loader:
            x, soft = x.to(device, non_blocking=True), soft.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(x)
            loss = criterion(out, soft)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(hard.numpy())
    return (total / len(loader.dataset), accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


@torch.no_grad()
def predict_paths(model, paths, transform, device, bs=64):
    model.eval()
    probs = []
    for i in range(0, len(paths), bs):
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
        probs.append(F.softmax(model(x), 1).cpu())
    return torch.cat(probs).numpy() if probs else np.zeros((0, NUM_CLASSES))


def save_gradcam_examples(model, paths, preds, labels, out_dir, device, n=5, seed=42):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    _, eval_tf = build_transforms()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    ok, bad = np.where(preds == labels)[0], np.where(preds != labels)[0]
    groups = {
        "correct": rng.choice(ok, min(n, len(ok)), replace=False) if len(ok) else [],
        "incorrect": rng.choice(bad, min(n, len(bad)), replace=False) if len(bad) else [],
        "per_class": [rng.choice(np.where(labels == c)[0]) for c in range(NUM_CLASSES) if (labels == c).any()],
    }
    for g, idxs in groups.items():
        for i in idxs:
            img = Image.open(paths[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            heat = cam(input_tensor=eval_tf(img).unsqueeze(0).to(device),
                       targets=[ClassifierOutputTarget(int(preds[i]))])[0]
            overlay = show_cam_on_image(np.asarray(img, dtype=np.float32) / 255.0, heat, use_rgb=True)
            Image.fromarray(overlay).save(os.path.join(
                out_dir, f"{g}_{i}_true-{IDX_TO_EMOTION[labels[i]]}_pred-{IDX_TO_EMOTION[preds[i]]}.jpg"))
    print(f"[OK] Grad-CAM images saved to {out_dir}")


# ───────────────────────── Main ─────────────────────────

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"resnet50_vggface2_rafau_{args.imbalance}_{args.labels}"
    print(f"Device: {device} | imbalance: {args.imbalance} | labels: {args.labels}")

    items = load_rafau_with_emotions(args.rafau_dir, args.rafml_dir)
    test_scored, test_neutral = load_rafdb_test(args.rafdb_test_dir)
    items, n_removed = remove_test_overlap(
        items, [p for p, _ in test_scored] + test_neutral,
        os.path.join(args.out_dir, "leakage_rafau_vs_rafdb_test.csv"), args.dup_threshold)

    hard = np.array([it["hard"] for it in items])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(items)), hard))

    def to_tuples(idx, soft):
        return [(items[i]["path"], items[i]["soft"] if soft else None, items[i]["hard"]) for i in idx]

    train_tf, eval_tf = build_transforms()
    train_ds = ImageDataset(to_tuples(tr_idx, args.labels == "soft"), train_tf)
    val_ds = ImageDataset(to_tuples(va_idx, False), eval_tf)   # val always scored on argmax label
    counts = np.bincount(hard[tr_idx], minlength=NUM_CLASSES)
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")
    for i, c in enumerate(counts):
        print(f"  {EMOTION_NAMES[i]:>10s}: {c}")

    if args.imbalance == "weighted_ce":
        w = 1.0 / counts
        w = torch.tensor(w / w.sum() * NUM_CLASSES, dtype=torch.float32)
        criterion = SoftTargetCE(w).to(device)
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    else:
        sw = 1.0 / counts[hard[tr_idx]]
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        criterion = SoftTargetCE().to(device)
        train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_criterion = SoftTargetCE().to(device)

    model = build_model(args.weights).to(device)
    head_ids = {id(p) for p in model.fc.parameters()}
    optimizer = optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": list(model.fc.parameters()), "lr": args.lr * 10}], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_path = os.path.join(args.out_dir, f"best_{tag}.pth")
    history, best_f1, best_epoch, bad = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va = run_epoch(model, val_loader, val_criterion, device)
        scheduler.step(va[2])
        history.append(dict(epoch=epoch, train_loss=tr[0], train_acc=tr[1], train_f1=tr[2],
                            val_loss=va[0], val_acc=va[1], val_f1=va[2]))
        print(f"Ep {epoch:02d} {time.time()-t0:5.0f}s | lr {optimizer.param_groups[0]['lr']:.1e} | "
              f"train loss {tr[0]:.4f} acc {tr[1]:.4f} F1 {tr[2]:.4f} | val loss {va[0]:.4f} acc {va[1]:.4f} F1 {va[2]:.4f}")
        if va[2] > best_f1:
            best_f1, best_epoch, bad = va[2], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": va[2], "val_acc": va[1], "class_mapping": IDX_TO_EMOTION,
                        "config": {"architecture": "ResNet-50 (VGGFace2, Caffe-style)",
                                   "weights_source": os.path.basename(args.weights),
                                   "train_data": "RAF-AU images + RAF-ML emotion distributions",
                                   "labels": args.labels, "imbalance": args.imbalance,
                                   "removed_test_duplicates": n_removed,
                                   "img_size": IMG_SIZE, "mean_bgr": VGGFACE2_MEAN_BGR,
                                   "num_classes": NUM_CLASSES, "lr": args.lr,
                                   "batch_size": args.batch_size, "seed": args.seed}}, ckpt_path)
            print(f"   >>> saved best (val Macro F1 {va[2]:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Single cross-dataset test on RAF-DB ──
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    paths = [p for p, _ in test_scored]
    labels = np.array([y for _, y in test_scored])
    preds = predict_paths(model, paths, eval_tf, device).argmax(1)
    per_f1 = f1_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    neutral_preds = predict_paths(model, test_neutral, eval_tf, device).argmax(1)
    results = {
        "tag": tag, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "train_images": len(train_ds), "val_images": len(val_ds), "removed_test_duplicates": n_removed,
        "test_set": "RAF-DB basic test, folders 1-6", "test_images": int(len(labels)),
        "test_accuracy": accuracy_score(labels, preds),
        "test_macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "test_macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "test_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "per_class_f1": {EMOTION_NAMES[i]: float(v) for i, v in enumerate(per_f1)},
        "per_class_recall": {EMOTION_NAMES[i]: float(v) for i, v in enumerate(
            recall_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0))},
        "worst_class": EMOTION_NAMES[int(per_f1.argmin())], "worst_class_f1": float(per_f1.min()),
        "fairness_gap_f1": float(per_f1.max() - per_f1.min()),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
        "neutral_not_scored": {"count": len(test_neutral),
                               "predicted_as": {EMOTION_NAMES[k]: v for k, v in Counter(neutral_preds.tolist()).items()}},
        "history": history,
    }
    print(classification_report(labels, preds, labels=list(range(NUM_CLASSES)),
                                target_names=EMOTION_NAMES, digits=4, zero_division=0))
    print(f"RAF-DB TEST (6-class) acc {results['test_accuracy']:.4f} | macro F1 {results['test_macro_f1']:.4f} | "
          f"worst {results['worst_class']} F1 {results['worst_class_f1']:.4f} | gap {results['fairness_gap_f1']:.4f}")
    print(f"Neutral (not scored) predicted as: {results['neutral_not_scored']['predicted_as']}")
    with open(os.path.join(args.out_dir, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    if not args.no_gradcam:
        save_gradcam_examples(model, paths, preds, labels, os.path.join(args.out_dir, f"gradcam_{tag}"),
                              device, seed=args.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rafau_dir", required=True, help="RAF-AU folder containing RAFAU_label.txt and aligned/")
    p.add_argument("--rafml_dir", required=True, help="RAF-ML folder containing EmoLabel/distribution.txt")
    p.add_argument("--rafdb_test_dir", required=True, help="RAF-DB test folder with subfolders 1-7")
    p.add_argument("--weights", required=True, help="Path to resnet50_ft_weight.pkl")
    p.add_argument("--imbalance", choices=["weighted_ce", "sampler"], default="weighted_ce")
    p.add_argument("--labels", choices=["soft", "hard"], default="soft",
                   help="soft = RAF-ML crowd distribution (default), hard = argmax only")
    p.add_argument("--dup_threshold", type=float, default=0.93)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="models")
    p.add_argument("--no_gradcam", action="store_true")
    main(p.parse_args())
```

---

## 4. `app.py` (REPLACE)

```python
"""
FastAPI app: ResNet-50 + VGGFace2 emotion recognition (trained on RAF-AU/RAF-ML, tested on RAF-DB) with Grad-CAM.
6 classes: Surprise, Fear, Disgust, Happiness, Sadness, Anger. RAF-DB Neutral (folder 7) is reported, not scored.
Configure via environment variables (no hard-coded D:\\ paths):
  MODEL_PATH  (default models/best_resnet50_vggface2_rafau_weighted_ce_soft.pth)
  TEST_DIR    (default TestDb/DATASET/test)
"""
import io
import os
import base64
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from torchvision import transforms
from fpdf import FPDF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, f1_score

from collections import Counter
from vggface2_resnet import (build_model, VGGFace2Normalize, IDX_TO_EMOTION, EMOTION_NAMES, IMG_SIZE,
                             NUM_CLASSES, RAFDB_FOLDER_TO_IDX, RAFDB_NEUTRAL_FOLDER)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "best_resnet50_vggface2_rafau_weighted_ce_soft.pth"))
TEST_DIR = os.environ.get("TEST_DIR", os.path.join("TestDb", "DATASET", "test"))
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model: fail loudly, never serve random weights ──
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Trained checkpoint not found at {MODEL_PATH}. Run train_resnet50.py first.")
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = build_model(pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])  # strict=True
model = model.to(device).eval()
CFG = ckpt.get("config", {})
print(f"[OK] Loaded {CFG.get('architecture', 'ResNet-50')} | weights {CFG.get('weights_source')} | "
      f"val Macro F1 {ckpt.get('val_macro_f1', float('nan')):.4f}")

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    VGGFace2Normalize(),
])


def run_inference(image: Image.Image, with_cam: bool = True):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    ranking = sorted(({"emotion": IDX_TO_EMOTION[i], "confidence": float(p)} for i, p in enumerate(probs)),
                     key=lambda d: d["confidence"], reverse=True)
    cam_pil = None
    if with_cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
        rgb = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        cam_pil = Image.fromarray(show_cam_on_image(rgb, heat, use_rgb=True))
    return IDX_TO_EMOTION[pred_idx], float(probs[pred_idx]), ranking, cam_pil


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def read_index():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image file")
    emotion, conf, ranking, cam_pil = run_inference(image)
    return {
        "predicted_emotion": emotion,
        "confidence": conf,
        "predictions": ranking,          # all 6 classes, sorted
        "grad_cam": pil_to_b64(cam_pil),
    }


def list_test_images():
    if not os.path.isdir(TEST_DIR):
        raise HTTPException(status_code=404, detail=f"Test directory not found: {TEST_DIR}")
    items, neutral = [], []
    for folder in sorted(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, folder)
        if not os.path.isdir(path):
            continue
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(VALID_EXT)]
        if folder in RAFDB_FOLDER_TO_IDX:
            items += [(p, RAFDB_FOLDER_TO_IDX[folder]) for p in files]
        elif folder == RAFDB_NEUTRAL_FOLDER:
            neutral += files
    return items, neutral


@app.post("/generate-report")
def generate_report(samples_per_class: int = 5, seed: int = 42):
    """Metrics on the FULL RAF-DB test set (folders 1-6) + seeded Grad-CAM examples per class.
    Sync def -> runs in threadpool."""
    items, neutral = list_test_images()
    labels, preds = [], []
    for path, y in items:
        emo, _, _, _ = run_inference(Image.open(path), with_cam=False)
        labels.append(y)
        preds.append(EMOTION_NAMES.index(emo))
    labels, preds = np.array(labels), np.array(preds)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_f1 = f1_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    neutral_spread = Counter(run_inference(Image.open(p), with_cam=False)[0] for p in neutral)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 18, "Emotion AI - Test Set Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, "ResNet-50 (VGGFace2) | train RAF-AU+RAF-ML | test RAF-DB (6 classes) | Grad-CAM",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, f"Scored test images (RAF-DB folders 1-6): {len(labels)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Accuracy: {acc*100:.2f}%   Macro F1: {macro_f1:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Worst class: {EMOTION_NAMES[int(per_f1.argmin())]} (F1 {per_f1.min():.4f})   "
                   f"Fairness gap (best-worst F1): {per_f1.max()-per_f1.min():.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Per-class F1", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    for i, name in enumerate(EMOTION_NAMES):
        pdf.cell(0, 6, f"  {name}: {per_f1[i]:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.multi_cell(0, 6, f"RAF-DB Neutral (folder 7, {len(neutral)} images) is not a training class and is not "
                         f"scored. Predicted as: " + ", ".join(f"{k} {v}" for k, v in neutral_spread.most_common()))

    rng = random.Random(seed)
    for c in range(NUM_CLASSES):
        cls_items = [p for p, y in items if y == c]
        chosen = rng.sample(cls_items, min(samples_per_class, len(cls_items)))
        if not chosen:
            continue
        pdf.add_page()
        pdf.set_font("helvetica", "B", 15)
        pdf.cell(0, 10, f"Ground truth: {EMOTION_NAMES[c]} (RAF-DB folder {c+1})", new_x="LMARGIN", new_y="NEXT", align="C")
        for path in chosen:
            img = Image.open(path).convert("RGB")
            emo, conf, _, cam_pil = run_inference(img)
            tmp = []
            for im in (img.resize((IMG_SIZE, IMG_SIZE)), cam_pil):
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                im.save(t.name, format="JPEG")
                t.close()
                tmp.append(t.name)
            y0 = pdf.get_y()
            if y0 > 240:
                pdf.add_page()
                y0 = pdf.get_y()
            pdf.image(tmp[0], x=10, y=y0, w=35)
            pdf.image(tmp[1], x=48, y=y0, w=35)
            pdf.set_xy(88, y0)
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(0, 6, f"{os.path.basename(path)}\nPredicted: {emo} ({conf*100:.1f}%)\n"
                                 f"Ground truth: {EMOTION_NAMES[c]}\n"
                                 f"Result: {'CORRECT' if emo == EMOTION_NAMES[c] else 'MISMATCH'}")
            pdf.set_y(y0 + 40)
            for t in tmp:
                os.remove(t)

    out = os.path.join(os.getcwd(), "TestDb_Report.pdf")
    pdf.output(out)
    return FileResponse(out, media_type="application/pdf", filename="TestDb_Report.pdf")
```

---

## 5. `static/index.html`: response format changed

`/predict` returns:
```json
{"predicted_emotion": "Happiness", "confidence": 0.91,
 "predictions": [{"emotion": "Happiness", "confidence": 0.91}, "... 6 entries ..."],
 "grad_cam": "<base64 jpeg>"}
```
Replace any `p.au` / `top_au` usage:
```js
list.innerHTML = data.predictions.map(p =>
  `<li>${p.emotion}: ${(p.confidence*100).toFixed(1)}%</li>`).join("");
badge.textContent = `${data.predicted_emotion} (${(data.confidence*100).toFixed(1)}%)`;
```
Optional UI note: "Model predicts 6 emotions; Neutral is not a trained class."

---

## 6. `notebook_code.py` patches (ResNet-18 in-domain baseline, kept for reference)

```python
# 2c — crashes on CUDA as written
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 3g — label map
LABEL_TO_EMOTION = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness",
                    5: "Sadness", 6: "Anger", 7: "Neutral"}

# 4b text: Fear and Disgust are the minority classes (not Contempt)

# 7c / 12a — drop verbose (removed in recent PyTorch)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer,     mode="max", factor=0.5, patience=3)
mit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mit_optimizer, mode="max", factor=0.5, patience=3)

# 15a — stop adding new Grad-CAM hooks on every call
_GC_CACHE = {}
def _get_gradcam(model):
    if id(model) not in _GC_CACHE:
        _GC_CACHE[id(model)] = GradCAM(model, model.layer4[-1])
    return _GC_CACHE[id(model)]
#   inside predict_emotion:  gc = _get_gradcam(model)
```
Re-run the notebook after the label fix and re-confirm 83.67% / worst-class F1 under the correct class names.

---

## 7. PPT edits

| Slide | Change |
|---|---|
| 1 subtitle | "…ResNet-50 + VGGFace2, trained on RAF-AU, evaluated cross-dataset on RAF-DB" |
| 3 Abstract | Training data = RAF-AU images with RAF-ML crowd-sourced emotion distributions (6 emotions); test = RAF-DB. Drop "Contempt" |
| 4, 5, 11, 17 | Minority classes = **Fear, Disgust**; remove Contempt |
| 6 Objective | Add "evaluate cross-dataset generalisation (RAF-AU → RAF-DB)" and "train on soft (distribution) labels" |
| 12 Proposed system | ResNet-50, VGGFace2 weights, 6-class head, soft-label CE, duplicate removal between train and test, Neutral reported not scored |
| 14 Benefits | Replace the copy of Slide 12 with real benefits |
| 15 Feasibility | ~23.5M backbone params; 4.6k training images → ~2–4 min/epoch on a mid-range GPU |
| 18 / new results slide | Columns: ResNet-18 ImageNet (in-domain, 7-class) · ResNet-50 VGGFace2 weighted CE (cross-dataset, 6-class) · ResNet-50 VGGFace2 sampler. Rows: Accuracy, Macro F1, worst-class F1, fairness gap. **Footnote the protocol difference** |
| New limitation bullet | "Neutral cannot be evaluated: RAF-AU/RAF-ML has no Neutral class. ~2% RAF-AU↔RAF-DB duplicates were found and removed." |

---

## 8. README "How it works" replacement

```markdown
## Data protocol
- Train: RAF-AU aligned images (4,601). RAF-AU provides only AU labels; the same images are in RAF-ML,
  whose crowd-sourced 6-emotion distributions (Surprise, Fear, Disgust, Happiness, Sadness, Anger)
  are used as targets. RAF-AU images that duplicate RAF-DB test images are removed.
- Validation: stratified 15% of RAF-AU.
- Test: RAF-DB basic test set, folders 1–6 (cross-dataset). Neutral (folder 7) has no training class.

## Model
ResNet-50 with official VGGFace2 weights (Cao et al., 2018; PyTorch port by cydonia999), 6-class head.
Input 224×224, BGR, mean-subtracted.

## Training
Soft-label cross-entropy; weighted-CE baseline vs WeightedRandomSampler mitigation; AdamW
(backbone 1e-4, head 1e-3); ReduceLROnPlateau on val Macro F1; early stopping.

## App
/predict → emotion + probabilities + Grad-CAM (layer4[-1]).
/generate-report → PDF: full RAF-DB test metrics, per-class F1, fairness gap, Neutral prediction spread,
seeded Grad-CAM examples.
```

---

## 9. Run order (from `D:\Research4`)

```powershell
$AU = "RAF-AU-20261003T122554Z-1-001\RAF-AU"
$ML = "RAF-ML-20261003T121432Z-1-001\RAF-ML"
$DB = "TestDb\DATASET\test"
$W  = "weights\resnet50_ft_weight.pkl"

# 1. Baseline
.\venv\Scripts\python.exe train_resnet50.py --rafau_dir $AU --rafml_dir $ML --rafdb_test_dir $DB --weights $W --imbalance weighted_ce

# 2. Mitigation
.\venv\Scripts\python.exe train_resnet50.py --rafau_dir $AU --rafml_dir $ML --rafdb_test_dir $DB --weights $W --imbalance sampler

# (optional ablation) hard argmax labels instead of soft distributions
.\venv\Scripts\python.exe train_resnet50.py --rafau_dir $AU --rafml_dir $ML --rafdb_test_dir $DB --weights $W --labels hard

# 3. App
$env:MODEL_PATH = "models\best_resnet50_vggface2_rafau_weighted_ce_soft.pth"
$env:TEST_DIR   = $DB
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```

---

## 10. Verification checklist

- [ ] `RAF-AU: 4601 images with emotion labels (missing image: 0, missing RAF-ML label: 0)`
- [ ] `RAF-DB test: 2388 scored images (folders 1-6), 680 Neutral (folder 7, not scored)`
- [ ] Leakage line shows a small number removed (expected around 1–3% of RAF-AU). Open `models\leakage_rafau_vs_rafdb_test.csv` and eyeball the rows with similarity 0.90–0.93 (logged, not removed)
- [ ] `[OK] VGGFace2 weights loaded: 265/265 tensors`
- [ ] Both `results_*weighted_ce_soft.json` and `results_*sampler_soft.json` exist; compare `test_macro_f1`, `worst_class_f1`, `fairness_gap_f1`
- [ ] Grad-CAM focuses on eyes, brows and mouth
- [ ] App refuses to start without a checkpoint
- [ ] No "Contempt" class left in the notebook, app or PPT: `Select-String -Path notebook_code.py,app.py,static\index.html -Pattern Contempt` returns nothing (the "NOT Contempt" comment in vggface2_resnet.py is intentional)
