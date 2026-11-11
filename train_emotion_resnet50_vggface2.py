"""
Emotion classifier: ResNet-50 initialised with VGGFace2 weights, fine-tuned on RAF-DB.

Why this script exists
----------------------
The usual reasons a "ResNet-50 + VGGFace2" FER model gets stuck at 30-40% are:
  1. The backbone is never actually given VGGFace2 weights (weights=None, or
     load_state_dict(strict=False) silently skipping every key).
  2. The VGGFace2 ResNet-50 weights (converted from Caffe) are loaded into
     torchvision's resnet50. Keys match, but the Caffe model puts the stride-2
     on the first 1x1 conv of each bottleneck and uses a ceil_mode max-pool, so
     the features come out wrong.
  3. Wrong input format. The VGGFace2 Caffe weights expect BGR pixels in the
     0-255 range minus a per-channel mean, NOT ImageNet mean/std on 0-1 RGB.
This script fixes all three and refuses to train if the weights did not load.

Weights
-------
Use resnet50_ft_weight.pkl (ResNet-50 pre-trained on MS1M, fine-tuned on
VGGFace2) from https://github.com/cydonia999/VGGFace2-pytorch (README, "Pretrained
models"). The network definition below follows that repository (MIT licence).

Usage
-----
python train_emotion_resnet50_vggface2.py \
    --data_dir "D:/Research4/TestDb/DATASET" \
    --weights  "D:/Research4/models/resnet50_ft_weight.pkl" \
    --out      "D:/Research4/models/best_resnet50_vggface2_emotion.pth"

The data folder must contain train/1..7 and test/1..7 (RAF-DB convention:
1 Surprise, 2 Fear, 3 Disgust, 4 Happiness, 5 Sadness, 6 Anger, 7 Neutral).
"""
import argparse
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
MEAN_BGR = torch.tensor([91.4953, 103.8827, 131.0912]).view(3, 1, 1)  # resnet50_ft.prototxt


# --------------------------------------------------------------------------
# VGGFace2 ResNet-50 (Caffe layout: stride on conv1 of the bottleneck, ceil_mode max-pool)
# --------------------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class VGGFace2ResNet50(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)   # Grad-CAM target layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2048, num_classes)          # new emotion head ("fc" is the 8631-way identity head, not used)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.head(self.dropout(self.features(x)))


def load_vggface2_weights(model, path):
    """Load VGGFace2 weights and FAIL LOUDLY if they do not cover the backbone."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            src = pickle.load(f, encoding="latin1")
        src = {k: torch.from_numpy(v) for k, v in src.items()}
    else:
        src = torch.load(path, map_location="cpu")
        src = src.get("state_dict", src)
        src = {k.replace("module.", ""): v for k, v in src.items()}
    own = model.state_dict()
    # num_batches_tracked buffers are not stored in Caffe conversions; they do not matter.
    backbone_keys = [k for k in own if not k.startswith("head.") and not k.endswith("num_batches_tracked")]
    loaded, shape_bad = 0, []
    for k in backbone_keys:
        if k in src:
            if src[k].shape == own[k].shape:
                own[k].copy_(src[k]); loaded += 1
            else:
                shape_bad.append(k)
    model.load_state_dict(own)
    real_missing = [k for k in backbone_keys if k not in src]
    print(f"[weights] loaded {loaded}/{len(backbone_keys)} backbone tensors "
          f"(missing {len(real_missing)}, shape mismatch {len(shape_bad)})")
    if real_missing or shape_bad:
        print("  first missing:", real_missing[:5], " first shape mismatch:", shape_bad[:5])
        raise SystemExit("VGGFace2 weights did not load into the backbone - stopping. "
                         "Use resnet50_ft_weight.pkl from cydonia999/VGGFace2-pytorch.")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class ToVGGFace2Tensor:
    """PIL RGB -> float tensor in BGR 0-255 minus VGGFace2 mean (what the Caffe weights expect)."""
    def __call__(self, img):
        t = transforms.functional.pil_to_tensor(img).float()   # RGB, 0-255
        return t[[2, 1, 0]] - MEAN_BGR


def build_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        ToVGGFace2Tensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])
    test_tf = transforms.Compose([transforms.Resize((size, size)), ToVGGFace2Tensor()])
    return train_tf, test_tf


def list_images(root):
    items = []
    for k in range(1, 8):
        d = os.path.join(root, str(k))
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, f), k - 1))   # folder 1..7 -> class 0..6
    return items


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def _macro_f1(ys, ps, num_classes=7):
    f1s = []
    for k in range(num_classes):
        tp = int(((ys == k) & (ps == k)).sum())
        fp = int(((ys != k) & (ps == k)).sum())
        fn = int(((ys == k) & (ps != k)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def _stratified_split(labels, val_frac, seed):
    rng = np.random.RandomState(seed)
    tr_idx, va_idx = [], []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.extend(idx[:n_val].tolist())
        tr_idx.extend(idx[n_val:].tolist())
    return np.array(tr_idx), np.array(va_idx)


def evaluate(model, loader, device, tta=True):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))   # horizontal-flip test-time augmentation
            ps += logits.argmax(1).cpu().tolist(); ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    return (ys == ps).mean(), _macro_f1(ys, ps), ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="models/best_resnet50_vggface2_emotion.pth")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup_epochs", type=int, default=2, help="head-only epochs before unfreezing")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate (head gets 10x)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight_power", type=float, default=0.5, help="0 = no weighting, 1 = full inverse frequency")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--select_metric", choices=["macro_f1", "accuracy"], default="macro_f1",
                    help="validation metric for choosing the best epoch; 'accuracy' + --class_weight_power 0 "
                         "= baseline WITHOUT fairness mitigation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="training randomness (init of head, shuffling, augmentation)")
    ap.add_argument("--split_seed", type=int, default=42, help="keep at 42 so every model shares the same validation split")
    ap.add_argument("--resume", action="store_true", help="continue from <out>_last.pth after a crash")
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate the best checkpoint on TEST")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--no_cudnn", action="store_true", help="disable cuDNN (slower; last resort)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, "| torch", torch.__version__, "| cuDNN", torch.backends.cudnn.version() if device.type == "cuda" else "-")
    if device.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = False
        if a.no_cudnn:
            torch.backends.cudnn.enabled = False
    use_amp = device.type == "cuda" and not a.no_amp

    # data: stratified validation split from TRAIN; TEST is used once at the end
    train_all = list_images(os.path.join(a.data_dir, "train"))
    test_items = list_images(os.path.join(a.data_dir, "test"))
    labels = np.array([y for _, y in train_all])
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        tr_idx, va_idx = next(StratifiedShuffleSplit(1, test_size=a.val_frac, random_state=a.split_seed).split(labels, labels))
    except ImportError:
        tr_idx, va_idx = _stratified_split(labels, a.val_frac, a.split_seed)
    train_tf, test_tf = build_transforms(a.img_size)
    tr = ImageList([train_all[i] for i in tr_idx], train_tf)
    va = ImageList([train_all[i] for i in va_idx], test_tf)
    te = ImageList(test_items, test_tf)
    counts = np.bincount(labels[tr_idx], minlength=7)
    print("[data] train", len(tr), "val", len(va), "test", len(te))
    print("[data] train counts:", dict(zip(EMOTIONS, counts.tolist())))
    pin = device.type == "cuda"
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=pin, drop_last=True)
    vl = DataLoader(va, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=pin)
    tel = DataLoader(te, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=pin)

    # model
    model = VGGFace2ResNet50(num_classes=7)
    load_vggface2_weights(model, a.weights)
    model.to(device)

    w = (counts.sum() / (7.0 * counts)) ** a.class_weight_power
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("[loss] class weights:", [round(float(v), 3) for v in w])
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)

    backbone = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr}, {"params": model.head.parameters(), "lr": a.lr * 10}],
                            weight_decay=1e-4)
    steps_per_epoch = len(tl); total = a.epochs * steps_per_epoch; warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_ep, bad, start_ep = -1, 0, 0, 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    last_path = os.path.splitext(a.out)[0] + "_last.pth"
    if a.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best_f1, best_ep, bad, start_ep = st["best_f1"], st["best_ep"], st["bad"], st["epoch"] + 1
        print(f"[resume] continuing from epoch {start_ep} (best val {a.select_metric} so far {best_f1*100:.2f}% at epoch {best_ep})")
    elif a.resume:
        print(f"[resume] {last_path} not found - starting from scratch")
    if a.eval_only:
        start_ep = a.epochs + 1
    oom_count = 0
    for ep in range(start_ep, a.epochs + 1):
        head_only = ep <= a.warmup_epochs
        for p in backbone:
            p.requires_grad = not head_only
        model.train(); t0 = time.time(); run_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    # Another program briefly took GPU memory: free the cache, skip this batch, carry on.
                    oom_count += 1
                    del e
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print(f"[warn] CUDA out of memory - skipped batch ({oom_count} so far). Close other GPU programs (e.g. the running app).")
                    if oom_count > 25:
                        raise SystemExit("Too many out-of-memory errors - rerun with --resume --batch_size 48")
                    time.sleep(2)
                    continue
                # Rare cuDNN failures (e.g. CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH) should not kill a long run:
                # skip this batch and continue with cuDNN disabled.
                if "cudnn" not in msg or not torch.backends.cudnn.enabled:
                    raise
                print(f"[warn] cuDNN error, skipping batch and disabling cuDNN for the rest of the run: {e}")
                torch.backends.cudnn.enabled = False
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                continue
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item() * len(y)
        acc, mf1, _, _ = evaluate(model, vl, device)
        print(f"epoch {ep:02d} {'(head only) ' if head_only else ''}loss {run_loss/len(tr):.4f} | "
              f"val acc {acc*100:.2f}% | val macro-F1 {mf1*100:.2f}% | {time.time()-t0:.0f}s")
        sel = mf1 if a.select_metric == "macro_f1" else acc
        if sel > best_f1:
            best_f1, best_ep, bad = sel, ep, 0
            torch.save({"state_dict": model.state_dict(), "classes": EMOTIONS, "img_size": a.img_size,
                        "preprocess": "RGB->BGR, 0-255, minus mean_bgr [91.4953, 103.8827, 131.0912]",
                        "val_macro_f1": mf1, "epoch": ep, "seed": a.seed, "split_seed": a.split_seed,
                        "val_frac": a.val_frac, "class_weight_power": a.class_weight_power,
                        "select_metric": a.select_metric,
                        "fairness_mitigation": a.class_weight_power > 0 or a.select_metric == "macro_f1"}, a.out)
        elif not head_only:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best_f1": best_f1, "best_ep": best_ep, "bad": bad}, last_path)
        if not head_only and bad >= a.patience:
            print("[early stop]"); break

    # final, one-time test evaluation with the best checkpoint
    best = torch.load(a.out, map_location=device)
    model.load_state_dict(best["state_dict"]); best_ep = best.get("epoch", best_ep)
    acc, mf1, ys, ps = evaluate(model, tel, device)
    print(f"\n[best epoch {best_ep}] TEST accuracy {acc*100:.2f}% | TEST macro-F1 {mf1*100:.2f}%")
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(ys, ps, target_names=EMOTIONS, digits=4, zero_division=0))
        print("confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(ys, ps))
    except ImportError:
        cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
        for t, p in zip(ys, ps):
            cm[t, p] += 1
        print("confusion matrix (rows = true, cols = predicted):")
        print(cm)


if __name__ == "__main__":
    main()
