from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import numpy as np
import base64
import random
import tempfile
from fpdf import FPDF
from pytorch_grad_cam import GradCAM, LayerCAM
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from train_emotion_resnet50_vggface2 import VGGFace2ResNet50, ToVGGFace2Tensor
from fairness import load_report, prediction_context, per_class_metrics, disparity_summary, compare_with_without

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── RAF-DB emotion label mapping (folder number → emotion name) ──
EMOTION_MAP = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}

AU_DESCRIPTIONS = {
    "1": "Inner Brow Raiser (Frontalis pars medialis)",
    "2": "Outer Brow Raiser (Frontalis pars lateralis)",
    "4": "Brow Lowerer (Corrugator supercilii / Depressor supercilii)",
    "5": "Upper Lid Raiser (Levator palpebrae superioris)",
    "6": "Cheek Raiser (Orbicularis oculi pars orbitalis)",
    "7": "Lid Tightener (Orbicularis oculi pars palpebralis)",
    "9": "Nose Wrinkler (Levator labii superioris alaeque nasi)",
    "10": "Upper Lip Raiser (Levator labii superioris)",
    "11": "Nasolabial Deepener (Zygomaticus minor)",
    "12": "Lip Corner Puller (Zygomaticus major - Duchenne Smile)",
    "13": "Sharp Lip Puller (Levator anguli oris)",
    "14": "Dimpler (Buccinator)",
    "15": "Lip Corner Depressor (Depressor anguli oris)",
    "16": "Lower Lip Depressor (Depressor labii inferioris)",
    "17": "Chin Raiser (Mentalis)",
    "18": "Lip Pucker (Incisivii labii superioris)",
    "19": "Tongue Show",
    "20": "Lip Stretcher (Risorius / Platysma)",
    "22": "Lip Funneler (Orbicularis oris)",
    "23": "Lip Tightener (Orbicularis oris)",
    "24": "Lip Pressor (Orbicularis oris)",
    "25": "Lips Part (Depressor labii inferioris)",
    "26": "Jaw Drop (Masseter & Temporal relaxed)",
    "27": "Mouth Stretch (Pterygoids & Digastric)",
    "28": "Lip Suck (Orbicularis oris)",
    "43": "Eyes Closed / Droop",
    "L12": "Left Lip Corner Puller (Unilateral Smirk)",
    "R12": "Right Lip Corner Puller (Unilateral Smirk)",
    "L14": "Left Dimpler",
    "R14": "Right Dimpler",
}

EMOTION_META = {
    "Happiness": {
        "emoji": "😊",
        "color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_border": "rgba(16, 185, 129, 0.4)",
        "description": "Genuine positive affect with Duchenne marker activation (cheek elevation + lip puller).",
        "focal_aus": "AU12 (Lip Corner Puller) + AU6 (Cheek Raiser)",
        "rule": "Ekman Duchenne Smile Rule"
    },
    "Surprise": {
        "emoji": "😲",
        "color": "#8b5cf6",
        "badge_bg": "rgba(139, 92, 246, 0.15)",
        "badge_border": "rgba(139, 92, 246, 0.4)",
        "description": "Startle reaction with high arched eyebrows, wide open eyes, and lowered mandible.",
        "focal_aus": "AU1+2 (Brow Raisers) + AU5 (Lid Raiser) + AU26/27 (Jaw Drop)",
        "rule": "Ekman Startle / Arched Brow Rule"
    },
    "Fear": {
        "emoji": "😨",
        "color": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.4)",
        "description": "Alarm response characterized by widened eyes, horizontal lip tension, and terror mouth stretch.",
        "focal_aus": "AU20 (Lip Stretcher) + AU1+2 + AU27/16 (Mouth Stretch)",
        "rule": "Terror & Lateral Retraction Rule"
    },
    "Sadness": {
        "emoji": "😢",
        "color": "#38bdf8",
        "badge_bg": "rgba(56, 189, 248, 0.15)",
        "badge_border": "rgba(56, 189, 248, 0.4)",
        "description": "Depressed affect with medial eyebrow knotting, furrowing, and downward lip depression.",
        "focal_aus": "AU1 (Inner Brow) + AU4 (Brow Lowerer) + AU15/17 (Lip Depressors)",
        "rule": "Medial Brow Knot / Commissure Depressor Rule"
    },
    "Disgust": {
        "emoji": "🤢",
        "color": "#14b8a6",
        "badge_bg": "rgba(20, 184, 166, 0.15)",
        "badge_border": "rgba(20, 184, 166, 0.4)",
        "description": "Aversion response driven by intense nasal wrinkling and upper labial retraction.",
        "focal_aus": "AU9 (Nose Wrinkler) + AU10 (Upper Lip Raiser)",
        "rule": "Nasal Wrinkle & Levator Labii Rule"
    },
    "Anger": {
        "emoji": "😡",
        "color": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.4)",
        "description": "Hostile confrontational expression with lowered corrugator brows and tightened or parted lips.",
        "focal_aus": "AU4 (Brow Lowerer) + AU23/24 (Lip Tightener/Pressor) + AU16",
        "rule": "Corrugator Brow Lowerer & Orbicularis Rule"
    },
    "Contempt": {
        "emoji": "😏",
        "color": "#ec4899",
        "badge_bg": "rgba(236, 72, 153, 0.15)",
        "badge_border": "rgba(236, 72, 153, 0.4)",
        "description": "Unilateral disdain or superiority characterized by asymmetric smirk or unilateral dimple.",
        "focal_aus": "L12/R12 (Unilateral Puller) or AU14 (Dimpler)",
        "rule": "Unilateral Facial Asymmetry Rule"
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#94a3b8",
        "badge_bg": "rgba(148, 163, 184, 0.15)",
        "badge_border": "rgba(148, 163, 184, 0.4)",
        "description": "Relaxed face with no strong facial muscle movement.",
        "focal_aus": "No dominant Action Unit (resting face)",
        "rule": "Resting Baseline"
    },
    "Neutral / Unknown": {
        "emoji": "😐",
        "color": "#94a3b8",
        "badge_bg": "rgba(148, 163, 184, 0.15)",
        "badge_border": "rgba(148, 163, 184, 0.4)",
        "description": "Relaxed facial musculature without significant action unit deformation.",
        "focal_aus": "Resting baseline",
        "rule": "Resting Baseline"
    }
}


# ── FACS-based AU → Emotion Classification Engine ──
# Uses Ekman FACS psychological ground truth, canonical AU normalization,
# Duchenne marker validation, and mutual emotion discrimination.

def canonical_au(raw_name: str) -> str:
    """Normalize AU labels like L1, R1, B22, T23, AD19 to canonical AU numbers."""
    import re
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def classify_emotion_from_aus(au_data, allow_contempt=True) -> str:
    """
    FACS Psychological Mapping Engine:
    Accepts either:
      - dict: {au_name: probability/confidence}
      - set/list: detected AU names
    Returns derived emotion category.
    """
    if isinstance(au_data, (set, list, tuple)):
        au_probs = {au: 1.0 for au in au_data}
    elif isinstance(au_data, dict):
        au_probs = au_data
    else:
        return "Neutral / Unknown"

    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))

    def p(c): return c_probs.get(str(c), 0.0)

    # Unilateral smirk / dimple detection for Contempt
    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)
    is_unilateral_smile = ((l12 > 0.4 and r12 < 0.25) or (r12 > 0.4 and l12 < 0.25) or (l12 > 0.55 and r12 < 0.35)) and p(6) < 0.5
    is_unilateral_dimple = (l14 > 0.4 and r14 < 0.25) or (r14 > 0.4 and l14 < 0.25)

    scores = {
        'Surprise': 0.0,
        'Fear': 0.0,
        'Disgust': 0.0,
        'Happiness': 0.0,
        'Sadness': 0.0,
        'Anger': 0.0,
        'Contempt': 0.0
    }

    # 1. SURPRISE: Arched brows (1+2), wide eyes (5), slack/open jaw (26/27)
    if p(1) > 0.4 and p(2) > 0.4 and p(5) > 0.4:
        scores['Surprise'] += (p(1) + p(2) + p(5)) * 2.5
        if p(26) > 0.4 or p(27) > 0.4:
            scores['Surprise'] += max(p(26), p(27)) * 2.5
    elif p(5) > 0.5 and (p(26) > 0.4 or p(27) > 0.4) and p(16) < 0.4:
        scores['Surprise'] += (p(5) + max(p(26), p(27))) * 2.2
        if p(1) > 0.4 or p(2) > 0.4:
            scores['Surprise'] += (p(1) + p(2)) * 1.5
    elif p(27) > 0.7 and p(16) < 0.4 and p(4) < 0.4 and p(9) < 0.4:
        scores['Surprise'] += p(27) * 4.0
    elif (p(1) > 0.5 and p(2) > 0.5) and (p(26) > 0.5 or p(27) > 0.5) and p(16) < 0.4:
        scores['Surprise'] += (p(1) + p(2)) * 2.0
    if p(19) > 0.5:
        scores['Surprise'] *= 0.1

    # 2. FEAR: Terror scream (27+16 with no roaring brow 4), lip stretcher (20+27/25)
    if p(27) > 0.7 and p(16) > 0.6:
        if (p(9) > 0.7 or p(10) > 0.85) and p(7) > 0.5 and p(1) < 0.4:
            scores['Anger'] += (p(27) + p(16) + max(p(9), p(10)) + p(7)) * 4.5
        elif p(9) < 0.6 or (p(20) > 0.5 and p(7) < 0.7) or p(4) < 0.5:
            scores['Fear'] += (p(27) + p(16)) * 6.5
            if p(1) > 0.4 or p(2) > 0.4:
                scores['Fear'] += 2.5
            if p(20) > 0.5:
                scores['Fear'] += p(20) * 3.0
            if p(22) > 0.5:
                scores['Fear'] += p(22) * 2.0
        else:
            scores['Anger'] += (p(27) + p(16) + p(9) + p(7)) * 4.0
    elif p(20) > 0.55 and p(27) > 0.6:
        scores['Fear'] += (p(20) + p(27)) * 5.5
        if p(5) > 0.4:
            scores['Fear'] += p(5) * 2.0
    elif (p(1) > 0.5 and p(2) > 0.5) and p(27) > 0.7:
        scores['Fear'] += (p(1) + p(2) + p(27)) * 3.0
    elif p(20) > 0.6 and p(16) > 0.5 and p(4) < 0.4:
        scores['Fear'] += (p(20) + p(16)) * 4.0

    # 3. HAPPINESS: Duchenne smile AU12 (lip puller) + AU6 (cheek raiser)
    if p(12) > 0.45:
        base_hap = p(12) * 3.5
        if p(6) > 0.4:
            base_hap += p(6) * 5.0
        if p(25) > 0.4:
            base_hap += p(25) * 1.5
        
        if p(27) > 0.7 and p(16) > 0.6:
            base_hap = 0.0
        elif p(4) > 0.6 and p(9) > 0.6 and p(20) > 0.6:
            base_hap *= 0.1
        # Crying sadness with high AU1 + AU4 + AU16 suppresses happiness
        elif p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.5:
            base_hap = 0.0
        elif is_unilateral_smile or (p(14) > 0.7 and p(6) < 0.5):
            base_hap *= 0.2
        scores['Happiness'] += base_hap

    if p(12) > 0.6 and p(25) > 0.4 and p(16) < 0.3 and p(27) < 0.3:
        scores['Happiness'] += (p(12) + p(25)) * 2.5
    elif p(12) > 0.4 and p(25) > 0.4 and p(6) > 0.3 and p(16) < 0.3 and p(27) < 0.3:
        scores['Happiness'] += (p(12) + p(25)) * 2.5

    # 4. SADNESS: Medial brow knot (1+4), weeping mouth (15+17, 16), or drooping eyelids (4+43)
    if p(1) > 0.4 and p(4) > 0.4 and p(27) < 0.6:
        scores['Sadness'] += (p(1) + p(4)) * 5.0
        if p(15) > 0.4 or p(17) > 0.4:
            scores['Sadness'] += 2.0
    # Intense weeping/crying grimace (AU1 + AU4 + AU16)
    if p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.6:
        scores['Sadness'] += (p(1) + p(4) + p(16)) * 6.0
    if p(15) > 0.5 and p(17) > 0.5 and p(27) < 0.6:
        scores['Sadness'] += (p(15) + p(17)) * 3.5
        if p(4) > 0.4:
            scores['Sadness'] += p(4) * 2.0
    elif p(4) > 0.5 and p(43) > 0.5 and p(12) < 0.4 and p(23) < 0.4 and p(24) < 0.4:
        scores['Sadness'] += (p(4) + p(43)) * 3.0
    # Suppress sadness only if AU12 is high WITHOUT AU1+AU4 weeping
    if p(12) > 0.7 and not (p(1) > 0.5 and p(4) > 0.6):
        scores['Sadness'] *= 0.05
    if p(27) > 0.7:
        scores['Sadness'] *= 0.1

    # 5. DISGUST: Nose wrinkler (9), upper lip raiser (10), tongue show (19)
    if p(19) > 0.5 and not is_unilateral_smile:
        scores['Disgust'] += p(19) * 8.0
    if p(9) > 0.5:
        scores['Disgust'] += p(9) * 4.5
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 3.5
    if p(10) > 0.45:
        scores['Disgust'] += p(10) * 3.0
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 2.5
    if p(4) > 0.6 and p(14) > 0.6 and p(9) > 0.6 and p(20) > 0.6:
        scores['Disgust'] += 15.0
    if p(4) > 0.6 and p(14) > 0.8 and p(12) > 0.8 and p(20) > 0.5:
        scores['Disgust'] += 15.0
    if p(4) > 0.4 and p(15) > 0.35 and p(17) > 0.35 and p(1) < 0.3 and p(23) < 0.3:
        scores['Disgust'] += (p(4) + p(15) + p(17)) * 3.5
    # AU10 nose/lip elevation without genuine smile AU6
    if p(10) > 0.45 and p(12) < 0.6 and p(6) < 0.3 and p(1) < 0.5:
        scores['Disgust'] += p(10) * 6.0
    # Duchenne laugh suppresses Disgust
    if p(12) > 0.8 and p(6) > 0.8 and p(27) < 0.6 and p(20) < 0.5 and not (p(1) > 0.5 and p(4) > 0.6):
        scores['Disgust'] *= 0.05
    # Inner brow raise AU1 suppresses Disgust (Disgust doesn't raise inner brow)
    if p(1) > 0.5 and p(9) < 0.6:
        scores['Disgust'] *= 0.2

    # 6. ANGER: Brow furrow (4) + lip tightener/pressor (23/24) or snarl roaring
    if p(4) > 0.4 and (p(23) > 0.4 or p(24) > 0.4):
        scores['Anger'] += (p(4) + max(p(23), p(24))) * 4.0
        if p(15) > 0.4:
            scores['Anger'] += p(15) * 1.5
    if p(4) > 0.6 and p(1) < 0.35 and (p(15) > 0.6 or p(17) > 0.6):
        scores['Anger'] += (p(4) + max(p(15), p(17))) * 5.0
    if p(16) > 0.7 and (p(25) > 0.7 or p(27) > 0.7) and p(9) > 0.7 and p(7) > 0.6:
        scores['Anger'] += (p(16) + max(p(25), p(27)) + p(9) + p(7)) * 4.0
    if p(16) > 0.9 and p(10) > 0.85 and (p(25) > 0.8 or p(26) > 0.8 or p(27) > 0.7) and p(1) < 0.45:
        scores['Anger'] += (p(16) + p(10) + max(p(25), p(26), p(27))) * 5.5
    if p(4) > 0.6 and p(7) > 0.5 and p(12) < 0.5:
        scores['Anger'] += (p(4) + p(7)) * 2.5
    if p(12) > 0.6 and p(6) > 0.5 and p(4) < 0.4:
        scores['Anger'] *= 0.1

    # 7. CONTEMPT: Unilateral smirk (L12/R12 or L14/R14), or Dimpler (14) + tight lips (23/24)
    if is_unilateral_smile or is_unilateral_dimple:
        scores['Contempt'] += 7.0
    if p(14) > 0.5:
        if (p(23) > 0.4 or p(24) > 0.4 or p(4) > 0.4) and p(6) < 0.7:
            scores['Contempt'] += (p(14) + max(p(23), p(24), p(4))) * 3.5
        if p(12) > 0.5 and p(6) < 0.7:
            scores['Contempt'] += (p(14) + p(12)) * 3.0
    if p(24) > 0.6 and p(23) > 0.6 and p(14) > 0.5:
        scores['Contempt'] += (p(24) + p(23) + p(14)) * 6.0

    if not allow_contempt:  # RAF-DB has no Contempt class: compare only the 7 dataset emotions
        scores.pop('Contempt', None)
    best = max(scores, key=scores.get)
    if scores[best] < 0.4:
        return 'Neutral / Unknown'
    return best


def load_aus():
    # Preferred: the 76 AU names shipped with the repo (no dataset annotations needed)
    names_path = os.path.join(BASE_DIR, "models", "au_labels.json")
    if os.path.exists(names_path):
        import json
        with open(names_path) as fjs:
            return list(json.load(fjs)["labels"])
    # Fallback: rebuild the list from the RAF-AU annotation file (local dataset copy)
    labels = set()
    label_path = os.path.join(BASE_DIR, "RAFAU_label.txt")
    if not os.path.exists(label_path):
        label_path = os.path.join(BASE_DIR, "RAF-AU-20261003T122554Z-1-001", "RAF-AU", "RAFAU_label.txt")
    if not os.path.exists(label_path):
        label_path = r"D:\Research4\RAF-AU-20261003T122554Z-1-001\RAF-AU\RAFAU_label.txt"
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1 and parts[1] != "null":
                    labels.update(parts[1].split('+'))
    except Exception as e:
        print("Could not load AU labels:", e)
    return sorted(list(labels))

all_aus = load_aus()
NUM_CLASSES = len(all_aus)

# 1. Action Unit Detection Model (RAF-AU trained VGGFace2 backbone)
print(f"Loading VGGFace2 AU Model (InceptionResnetV1) for {NUM_CLASSES} Action Units on {device}...")
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained=None, classify=True, num_classes=NUM_CLASSES)
try:
    au_model_path = os.path.join(BASE_DIR, "models", "best_vggface2_raf_au.pth")
    model.load_state_dict(torch.load(au_model_path, map_location=device))
    print("VGGFace2 AU Model weights loaded successfully!")
except Exception as e:
    print("Could not load AU model weights, you need to retrain using train_rafau.py!", e)

model = model.to(device)
model.eval()

# 2. Emotion Classification Model: ResNet-50 initialised with VGGFace2, fine-tuned on RAF-DB
#    (trained by train_emotion_resnet50_vggface2.py). Falls back to the older InceptionResnetV1
#    checkpoint only if the new one is missing.
EMOTION_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
emotion_model_path = os.path.join(BASE_DIR, "models", "best_resnet50_vggface2_emotion.pth")
ensemble_json = os.path.join(BASE_DIR, "models", "emotion_ensemble.json")
EMOTION_BACKBONE = None
emotion_cam = None
emotion_models = []                      # one or more ResNet-50 + VGGFace2 models (ensemble)
emotion_class_bias = torch.zeros(7)      # per-class log-prior offsets chosen on validation
emotion_paths = []
if os.path.exists(ensemble_json):
    import json
    with open(ensemble_json) as fjs:
        ens = json.load(fjs)
    emotion_paths = [p if os.path.isabs(p) else os.path.join(BASE_DIR, p) for p in ens["models"]]
    emotion_class_bias = torch.tensor(ens.get("class_bias", [0.0] * 7), dtype=torch.float32)
elif os.path.exists(emotion_model_path):
    emotion_paths = [emotion_model_path]
for pth in emotion_paths:
    ckpt = torch.load(pth, map_location=device)
    net = VGGFace2ResNet50(num_classes=7)
    net.load_state_dict(ckpt["state_dict"])
    emotion_models.append(net.to(device).eval())
    EMOTION_NAMES = list(ckpt.get("classes", EMOTION_NAMES))
    emotion_img_size = ckpt.get("img_size", 224)
    print(f"Loaded emotion model {os.path.basename(pth)} (epoch {ckpt.get('epoch')}, "
          f"val macro-F1 {ckpt.get('val_macro_f1', 0) * 100:.2f}%)")
if emotion_models:
    model_emotion = emotion_models[0]
    emotion_transform = transforms.Compose([
        transforms.Resize((emotion_img_size, emotion_img_size)),
        ToVGGFace2Tensor(),  # BGR, 0-255, minus VGGFace2 mean - same as training
    ])
    EMOTION_BACKBONE = "ResNet-50 + VGGFace2" + (f" (ensemble of {len(emotion_models)})" if len(emotion_models) > 1 else "")
    emotion_class_bias = emotion_class_bias.to(device)
else:
    print("Note: no ResNet-50 emotion model found - using the older InceptionResnetV1 emotion model.")
    model_emotion = InceptionResnetV1(pretrained=None, classify=True, num_classes=7)
    old_path = os.path.join(BASE_DIR, "models", "best_vggface2_emotion.pth")
    if os.path.exists(old_path):
        model_emotion.load_state_dict(torch.load(old_path, map_location=device))
        EMOTION_BACKBONE = "InceptionResnetV1 + VGGFace2 (old)"
    emotion_transform = None  # uses the shared ImageNet transform below

model_emotion = model_emotion.to(device)
model_emotion.eval()
# The UI and PDF use the AU heatmap only (emotion-model heatmaps were dropped: they collapsed onto the nose).

# Fairness audit of the deployed emotion model (written by `python fairness.py --data_dir TestDb/DATASET`)
FAIRNESS_JSON = os.path.join(BASE_DIR, "models", "fairness_report.json")
FAIRNESS_PNG = os.path.join(BASE_DIR, "models", "fairness_report.png")
FAIRNESS_REPORT = load_report(FAIRNESS_JSON)
print("Fairness report:", "loaded" if FAIRNESS_REPORT else "not found - run fairness.py to add test-set fairness metrics")

# Baseline WITHOUT fairness mitigation (no class weights, best epoch by accuracy) - trained by train_nofair.bat.
# Runs next to the deployed model so every prediction and the batch report can show "with vs without fairness".
BASELINE_PATH = os.path.join(BASE_DIR, "models", "emotion_r50_nofair.pth")
baseline_model, baseline_transform = None, None
if os.path.exists(BASELINE_PATH):
    _bck = torch.load(BASELINE_PATH, map_location=device)
    baseline_model = VGGFace2ResNet50(num_classes=7)
    baseline_model.load_state_dict(_bck["state_dict"])
    baseline_model = baseline_model.to(device).eval()
    _bsz = _bck.get("img_size", 224)
    baseline_transform = transforms.Compose([transforms.Resize((_bsz, _bsz)), ToVGGFace2Tensor()])
    print(f"Loaded baseline WITHOUT fairness: {os.path.basename(BASELINE_PATH)}")
else:
    print("Baseline without fairness not found - run train_nofair.bat to enable the with-vs-without comparison")


def baseline_predict(image):
    """Prediction of the baseline trained without fairness mitigation (flip TTA, no calibration)."""
    if baseline_model is None:
        return None
    x = baseline_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pb = (torch.softmax(baseline_model(x), 1) + torch.softmax(baseline_model(torch.flip(x, dims=[3])), 1))[0] / 2
    i = int(pb.argmax())
    return {"emotion": EMOTION_NAMES[i], "confidence": round(float(pb[i]), 4),
            "probabilities": {n: round(float(v), 4) for n, v in zip(EMOTION_NAMES, pb.cpu())}}


def _facs_name(name):
    return "Neutral" if str(name).startswith("Neutral") else name


def fairness_context(predicted_emotion, extras):
    """Per-prediction fairness / reliability context returned by /predict and used in the PDF report."""
    if extras.get("probs") is None:
        return {"available": False, "note": "Emotion came from the FACS rules only - no probability-based fairness context."}
    ctx = prediction_context(predicted_emotion, extras["probs"], FAIRNESS_REPORT,
                             _facs_name(extras.get("facs_emotion")), EMOTION_NAMES)
    ctx["available"] = True
    if FAIRNESS_REPORT:
        s = FAIRNESS_REPORT["final"]["summary"]
        ctx["model_audit"] = {k: s.get(k) for k in [
            "accuracy", "balanced_accuracy", "macro_f1", "worst_recall_class", "worst_recall", "best_recall_class",
            "best_recall", "recall_gap", "recall_ratio", "passes_four_fifths", "minority_classes",
            "minority_mean_recall", "majority_mean_recall", "minority_recall_gap"]}
        ctx["model_audit"]["n_test"] = FAIRNESS_REPORT.get("n_test")
        wv = FAIRNESS_REPORT.get("with_vs_without")
        if wv:
            ctx["model_audit"]["with_vs_without"] = {k: {kk: v[kk] for kk in ("without", "with", "change")}
                                                     for k, v in wv["metrics"].items()}
    b = extras.get("baseline")
    if b:
        ctx["without_fairness"] = {**b, "same_as_with_fairness": b["emotion"] == predicted_emotion}
    else:
        ctx["without_fairness"] = None
    return ctx


# TestDb test folder (used for demo samples and the batch report - never to look up answers)
TEST_DB_DIR = os.path.join(BASE_DIR, "TestDb", "DATASET", "test")
if not os.path.exists(TEST_DB_DIR):
    TEST_DB_DIR = r"D:\Research4\TestDb\DATASET\test"

# AU heatmap: LayerCAM over three depths of InceptionResnetV1 (block8 alone is a coarse 5x5 grid)
target_layers = [model.repeat_2[-1], model.mixed_7a, model.block8]
cam = LayerCAM(model=model, target_layers=target_layers)


def _border_taper(size=224, frac=0.08):
    r = np.ones(size); n = int(size * frac)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)); r[:n] = ramp; r[-n:] = ramp[::-1]
    return np.outer(r, r)


CAM_TAPER = _border_taper()


def refine_cam(maps):
    """Average raw CAMs (models / flipped views), smooth, robust-normalise and fade the image border.
    Removes the speckle and the min-max 'everything is red' effect of a single raw Grad-CAM."""
    g = np.mean(maps, 0).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 6)
    hi, lo = np.percentile(g, 99.5), np.percentile(g, 30)
    g = np.clip((g - lo) / (hi - lo + 1e-8), 0, 1)
    if g.shape == CAM_TAPER.shape:
        g = g * CAM_TAPER
    return g


def cam_overlay(image, gray):
    rgb = np.array(image.resize((gray.shape[1], gray.shape[0]))) / 255.0
    return Image.fromarray(show_cam_on_image(rgb, gray, use_rgb=True, image_weight=0.55))


def _b64(pil):
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def run_inference(image: Image.Image, filename: str = None):
    """
    Run the full pipeline on a PIL image:
    1. Action Unit prediction & confidence calculation via VGGFace2 RAF-AU model
    2. Deep emotion classification via VGGFace2 Emotion model
    3. Grad-CAM visual explainability heatmap on block8
    """
    tensor = transform(image).unsqueeze(0).to(device)

    # 1. Action Unit Predictions from VGGFace2 RAF-AU model
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    # Continuous AU probability map for FACS emotion classification
    au_probs = {all_aus[i]: float(probs[i]) for i in range(len(all_aus))}

    # 2. Emotion prediction - always from the model (no ground-truth lookup)
    emotion_confidence = None
    emotion_grad_cam_b64 = None
    extras = {"probs": None, "facs_emotion": classify_emotion_from_aus(au_probs, allow_contempt=False),
              "baseline": baseline_predict(image)}
    if EMOTION_BACKBONE is None:
        predicted_emotion = extras["facs_emotion"]
    else:
        em_tensor = emotion_transform(image).unsqueeze(0).to(device) if emotion_transform else tensor
        with torch.no_grad():
            if emotion_models:  # average softmax over models and flipped copies, then apply class offsets
                flipped = torch.flip(em_tensor, dims=[3])
                em_probs = torch.stack([torch.softmax(m(em_tensor), 1) + torch.softmax(m(flipped), 1)
                                        for m in emotion_models]).mean(0)[0] / 2
                em_probs = torch.softmax(torch.log(em_probs + 1e-9) + emotion_class_bias, dim=0)
            else:
                em_probs = torch.softmax(model_emotion(em_tensor), dim=1)[0]
        em_idx = int(em_probs.argmax())
        predicted_emotion = EMOTION_NAMES[em_idx]
        emotion_confidence = float(em_probs[em_idx])
        extras["probs"] = [float(v) for v in em_probs.detach().cpu()]

    # Collect detected AUs with confidence > 0.35
    predictions = []
    for i, prob in enumerate(probs):
        if prob > 0.35:
            predictions.append({"au": all_aus[i], "confidence": float(prob), "idx": i})
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    # Fallback to top predictions if none surpassed 0.35 threshold
    if not predictions:
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            predictions.append({"au": all_aus[idx], "confidence": float(probs[idx]), "idx": int(idx)})

    # Restrict to Top 5 AUs for clean frontend and PDF display
    display_predictions = predictions[:5]

    # Generate Grad-CAM for the top prediction
    grad_cam_b64 = None
    cam_pil = None
    top_au_name = None
    if display_predictions:
        top_idx = display_predictions[0]['idx']
        top_au_name = display_predictions[0]['au']
        targets = [ClassifierOutputTarget(top_idx)]
        maps = [cam(input_tensor=tensor, targets=targets)[0]]
        if not str(top_au_name)[:1] in ("L", "R"):  # left/right AUs swap sides when flipped - only flip symmetric ones
            maps.append(cam(input_tensor=torch.flip(tensor, dims=[3]), targets=targets)[0][:, ::-1])
        cam_pil = cam_overlay(image, refine_cam(maps))
        grad_cam_b64 = _b64(cam_pil)

    # Strip internal index before returning and attach anatomical description
    for p in display_predictions:
        p.pop("idx", None)
        au_str = str(p.get("au", ""))
        p["description"] = AU_DESCRIPTIONS.get(au_str, AU_DESCRIPTIONS.get(canonical_au(au_str), f"Action Unit {au_str}"))

    return (display_predictions, predicted_emotion, top_au_name, grad_cam_b64, cam_pil, emotion_confidence,
            emotion_grad_cam_b64, extras)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/samples")
async def get_samples():
    samples = []
    test_dir = TEST_DB_DIR
    if os.path.exists(test_dir):
        for f_num in sorted(os.listdir(test_dir)):
            f_path = os.path.join(test_dir, f_num)
            if os.path.isdir(f_path):
                emotion_name = EMOTION_MAP.get(f_num, f"Emotion {f_num}")
                meta = EMOTION_META.get(emotion_name, EMOTION_META["Neutral / Unknown"])
                images = [f for f in sorted(os.listdir(f_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    samples.append({
                        "folder": f_num,
                        "emotion": emotion_name,
                        "filename": images[0],
                        "url": f"/api/sample-image/{f_num}/{images[0]}",
                        "emoji": meta.get("emoji", "🎭"),
                        "color": meta.get("color", "#6366f1")
                    })
    return {"samples": samples}


@app.get("/api/sample-image/{folder}/{filename}")
async def get_sample_image(folder: str, filename: str):
    file_path = os.path.join(TEST_DB_DIR, folder, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    return HTMLResponse("Image not found", status_code=404)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        (predictions, predicted_emotion, top_au_name, grad_cam_b64, _, emotion_conf, emotion_cam_b64,
         extras) = run_inference(image, filename=file.filename)
        meta = EMOTION_META.get(predicted_emotion, EMOTION_META["Neutral / Unknown"])
        top_au_desc = AU_DESCRIPTIONS.get(str(top_au_name), AU_DESCRIPTIONS.get(canonical_au(str(top_au_name)), f"Action Unit {top_au_name}")) if top_au_name else None

        return {
            "predictions": predictions,
            "grad_cam": grad_cam_b64,
            "top_au": top_au_name,
            "top_au_description": top_au_desc,
            "predicted_emotion": predicted_emotion,
            "emotion_meta": meta,
            "emotion_confidence": emotion_conf,
            "emotion_grad_cam": emotion_cam_b64,
            "emotion_model": EMOTION_BACKBONE,
            "emotion_probabilities": ({n: round(p, 4) for n, p in zip(EMOTION_NAMES, extras["probs"])}
                                      if extras["probs"] else None),
            "fairness": fairness_context(predicted_emotion, extras),
            "without_fairness": extras.get("baseline"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": f"Model inference error: {str(e)}"})


@app.post("/generate-report")
async def generate_report():
    test_dir = TEST_DB_DIR
    if not os.path.exists(test_dir):
        return {"error": f"Directory not found: {test_dir}"}

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title page ──
    pdf.add_page()
    pdf.set_font("helvetica", "B", 22)
    pdf.cell(0, 20, "Emotion AI - Batch Test Report", ln=True, align='C')
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Emotion model: {EMOTION_BACKBONE} | AU model: VGGFace2 (RAF-AU) | Grad-CAM", ln=True, align='C')
    pdf.cell(0, 10, "All predictions come directly from the trained model", ln=True, align='C')
    pdf.ln(10)

    total_correct = 0
    total_images = 0
    emotion_stats = {}
    fair_rows = []   # per-image records for the fairness page

    all_class_folders = [f for f in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, f))]
    
    for class_folder in all_class_folders:
        class_path = os.path.join(test_dir, class_folder)
        ground_truth_emotion = EMOTION_MAP.get(class_folder, f"Unknown({class_folder})")
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        # Fully randomized selection of 5 images per emotion class on every run
        selected_images = random.sample(images, min(5, len(images)))

        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, f"Ground Truth: {ground_truth_emotion}  (Folder {class_folder})", ln=True, align='C')
        pdf.ln(5)

        class_correct = 0
        class_total = 0

        for idx, img_file in enumerate(selected_images):
            img_path = os.path.join(class_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            predictions, predicted_emotion, top_au, _, cam_pil, _, _, extras = run_inference(image, filename=img_file)
            fctx = fairness_context(predicted_emotion, extras)

            total_images += 1
            class_total += 1

            match = (predicted_emotion == ground_truth_emotion)
            if match:
                total_correct += 1
                class_correct += 1
            base = extras.get("baseline")
            fair_rows.append({"true": ground_truth_emotion, "pred": predicted_emotion, "correct": match,
                              "facs": _facs_name(extras.get("facs_emotion")),
                              "base_pred": base["emotion"] if base else None, **fctx})

            # Save temp images for PDF embedding
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
                image.resize((224, 224)).save(tmp_orig.name, format="JPEG")
                orig_tmp_path = tmp_orig.name

            cam_tmp_path = None
            if cam_pil is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_cam:
                    cam_pil.save(tmp_cam.name, format="JPEG")
                    cam_tmp_path = tmp_cam.name

            # ── Write to PDF ──
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 7, f"Image: {img_file}", ln=True)

            if pdf.get_y() > 240:
                pdf.add_page()
            y_before = pdf.get_y()
            pdf.image(orig_tmp_path, x=10, y=y_before, w=34)
            x_next = 46
            if cam_tmp_path:  # heatmap of the top Action Unit
                pdf.image(cam_tmp_path, x=46, y=y_before, w=34); x_next = 82
                pdf.set_font("helvetica", "", 6.5)
                pdf.set_xy(46, y_before + 34.5)
                pdf.cell(34, 3, f"AU {top_au} heatmap", align="C")

            pdf.set_xy(x_next + 2, y_before)
            pdf.set_font("helvetica", "", 8)
            det_aus = ', '.join([f"AU{p['au']}" for p in predictions]) if predictions else "None"
            match_str = "CORRECT" if match else "MISMATCH"
            info_text = (
                f"Predicted (with fairness): {predicted_emotion}\n"
                f"Ground Truth: {ground_truth_emotion}\n"
                f"Result: {match_str}\n"
                f"Top AU: {top_au or 'None'}\n"
                f"Detected AUs: {det_aus}"
            )
            if fctx.get("available"):
                info_text += (f"\nConfidence {fctx['confidence']*100:.0f}% | top-2 margin {fctx['margin']*100:.0f} pts | "
                              f"reliability {fctx['reliability']}{' | REVIEW' if fctx['needs_review'] else ''}\n"
                              f"FACS rules: {fctx.get('facs_rule_emotion')} "
                              f"({'agree' if fctx.get('facs_agreement') else 'disagree'})")
            if extras.get("baseline"):
                bp = extras["baseline"]["emotion"]
                info_text += (f"\nWithout fairness: {bp} ({'correct' if bp == ground_truth_emotion else 'wrong'})"
                              + ("" if bp == predicted_emotion else " - differs"))
            pdf.multi_cell(0, 4.1, info_text, align="L")

            pdf.ln(max(0, y_before + 41 - pdf.get_y()))

            os.remove(orig_tmp_path)
            if cam_tmp_path:
                os.remove(cam_tmp_path)

        emotion_stats[ground_truth_emotion] = (class_correct, class_total)

    # ── Summary page ──
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 15, "Evaluation Summary", ln=True, align='C')
    pdf.set_font("helvetica", "I", 10)
    pdf.cell(0, 7, "WITH fairness = deployed model. The without-fairness baseline is compared on the next page.", ln=True, align='C')
    pdf.set_font("helvetica", "", 12)
    accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    pdf.cell(0, 9, f"Total Images Tested: {total_images}", ln=True)
    pdf.cell(0, 9, f"Correct Predictions: {total_correct}", ln=True)
    pdf.cell(0, 9, f"Emotion Accuracy on this random sample (WITH fairness): {accuracy:.1f}%", ln=True)
    base_rows = [r for r in fair_rows if r.get("base_pred")]
    if base_rows:
        base_acc = sum(r["base_pred"] == r["true"] for r in base_rows) / len(base_rows) * 100
        pdf.cell(0, 9, f"Emotion Accuracy on the same images (WITHOUT fairness): {base_acc:.1f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "Per-Emotion Accuracy Breakdown:", ln=True)
    pdf.set_font("helvetica", "", 11)
    for emo, (c, n) in emotion_stats.items():
        pct = (c / n * 100) if n > 0 else 0
        pdf.cell(0, 7, f"  - {emo:12s}: {c}/{n} ({pct:.1f}%)", ln=True)

    add_fairness_page(pdf, fair_rows)

    pdf_path = os.path.join(BASE_DIR, "TestDb_Batch_Report.pdf")
    pdf.output(pdf_path)
    return FileResponse(pdf_path, media_type='application/pdf', filename="TestDb_Batch_Report.pdf")


def _ascii(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _para(pdf, h, text):
    pdf.multi_cell(0, h, text, new_x="LMARGIN", new_y="NEXT")


def _table(pdf, header, rows, widths, size=8):
    pdf.set_font("helvetica", "B", size)
    for h, w in zip(header, widths):
        pdf.cell(w, 6, _ascii(h), border=1, align="C")
    pdf.ln()
    pdf.set_font("helvetica", "", size)
    for r in rows:
        for v, w in zip(r, widths):
            pdf.cell(w, 5.5, _ascii(v), border=1, align="C")
        pdf.ln()


def _pct(v):
    return "-" if v is None else f"{v*100:.1f}%"


def add_fairness_page(pdf, rows):
    """Fairness & reliability pages of the batch report: this sample + the full test-set audit."""
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 14, "Fairness & Reliability", ln=True, align="C")
    pdf.set_font("helvetica", "", 9)
    _para(pdf, 5, _ascii(
        "RAF-DB has no demographic labels in this project, so fairness is measured across emotion classes: "
        "every true emotion should be recognised equally well (equal opportunity). Recall = share of true faces of a "
        "class that the model recognises. Recall ratio = worst / best recall; >= 0.80 passes the four-fifths rule."))
    pdf.ln(2)

    # ---- 1. this random sample ----
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, "1. This random sample", ln=True)
    names = EMOTION_NAMES
    idx = {n: i for i, n in enumerate(names)}
    y = [idx.get(r["true"], -1) for r in rows]
    p = [idx.get(r["pred"], -1) for r in rows]
    if rows:
        pcm = per_class_metrics(y, p, names)
        _table(pdf, ["Class", "Train share", "n", "Recall", "Precision", "F1", "Pred share"],
               [[r["class"], _pct(r["train_share"]), r["support"], _pct(r["recall"]), _pct(r["precision"]),
                 _pct(r["f1"]), _pct(r["pred_share"])] for r in pcm],
               [30, 24, 12, 22, 24, 22, 24])
        ds = disparity_summary(pcm, y, p)
        pdf.ln(2)
        pdf.set_font("helvetica", "", 9)
        if ds:
            lines = [
                f"Accuracy {_pct(ds['accuracy'])} | balanced accuracy (mean recall) {_pct(ds['balanced_accuracy'])}",
                f"Worst class {ds['worst_recall_class']} ({_pct(ds['worst_recall'])}) vs best {ds['best_recall_class']} "
                f"({_pct(ds['best_recall'])}): recall gap {ds['recall_gap']*100:.1f} points, ratio {ds['recall_ratio']:.2f} "
                f"({'passes' if ds['passes_four_fifths'] else 'fails'} the four-fifths rule)",
            ]
            if "minority_recall_gap" in ds:
                lines.append(f"Minority classes ({', '.join(ds['minority_classes'])}): mean recall "
                             f"{_pct(ds['minority_mean_recall'])} vs {_pct(ds['majority_mean_recall'])} for the others")
            for l in lines:
                _para(pdf, 5, _ascii(l))
        ctx = [r for r in rows if r.get("available")]
        if ctx:
            ok = [r["confidence"] for r in ctx if r["correct"]]
            bad = [r["confidence"] for r in ctx if not r["correct"]]
            flagged = [r for r in ctx if r["needs_review"]]
            facs_agree = sum(1 for r in ctx if r.get("facs_agreement"))
            pdf.ln(1)
            for l in [
                f"Mean confidence: {np.mean(ok)*100:.1f}% when correct" + (f", {np.mean(bad)*100:.1f}% when wrong" if bad else ""),
                f"Flagged for human review (low reliability or close call): {len(flagged)} of {len(ctx)}"
                f" - {sum(1 for r in flagged if not r['correct'])} of them were wrong; "
                f"{sum(1 for r in ctx if not r['correct'] and not r['needs_review'])} of the "
                f"{sum(1 for r in ctx if not r['correct'])} errors were not flagged",
                f"FACS rule engine agrees with the model on {facs_agree} of {len(ctx)} images "
                f"({facs_agree/len(ctx)*100:.0f}%) - an independent, muscle-based cross-check",
                f"Reliability bands: " + ", ".join(f"{b} {sum(1 for r in ctx if r['reliability']==b)}" for b in ["High", "Medium", "Low"]),
            ]:
                _para(pdf, 5, _ascii(l))
        pdf.set_font("helvetica", "I", 8)
        _para(pdf, 4.5, _ascii(f"Only {len(rows)} images (about 5 per class): a single error moves a class "
                                      "recall by 20 points. Use the full test-set audit below for conclusions."))
    pdf.ln(3)

    # ---- with vs without fairness on this sample ----
    _with_without_section(pdf, rows, y, p, idx)

    # ---- full test-set audit ----
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, "3. Full test-set audit (deployed model)", ln=True)
    if not FAIRNESS_REPORT:
        pdf.set_font("helvetica", "", 9)
        _para(pdf, 5, "Not available. Run:  venv\\Scripts\\python.exe fairness.py --data_dir TestDb/DATASET  "
                             "then restart the app.")
        return
    fr = FAIRNESS_REPORT
    _table(pdf, ["Class", "Train share", "n", "Recall", "Precision", "F1", "FNR", "FPR"],
           [[r["class"], _pct(r["train_share"]), r["support"], _pct(r["recall"]), _pct(r["precision"]),
             _pct(r["f1"]), _pct(r["fnr"]), _pct(r["fpr"])] for r in fr["final"]["per_class"]],
           [28, 22, 14, 21, 22, 21, 21, 21])
    s = fr["final"]["summary"]
    pdf.ln(2)
    pdf.set_font("helvetica", "", 9)
    for l in [
        f"{fr.get('n_test')} test images | accuracy {_pct(s.get('accuracy'))} | balanced accuracy {_pct(s['balanced_accuracy'])} "
        f"| macro-F1 {_pct(s['macro_f1'])}",
        f"Recall gap {s['recall_gap']*100:.1f} points ({s['worst_recall_class']} vs {s['best_recall_class']}), "
        f"ratio {s['recall_ratio']:.2f} - {'passes' if s['passes_four_fifths'] else 'fails'} the four-fifths rule",
    ] + ([f"Minority classes ({', '.join(s['minority_classes'])}): mean recall {_pct(s['minority_mean_recall'])} vs "
          f"{_pct(s['majority_mean_recall'])} (gap {s['minority_recall_gap']*100:.1f} points)"] if "minority_recall_gap" in s else []):
        _para(pdf, 5, _ascii(l))

    pdf.ln(2)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 8, "Mitigation comparison (class weighting, ensembling, per-class calibration)", ln=True)
    _table(pdf, ["Configuration", "Acc", "Bal. acc", "Worst class", "Recall gap", "Ratio"],
           [[c["name"][:62], _pct(c["summary"].get("accuracy")), _pct(c["summary"]["balanced_accuracy"]),
             f"{c['summary']['worst_recall_class']} {_pct(c['summary']['worst_recall'])}",
             f"{c['summary']['recall_gap']*100:.1f}", f"{c['summary']['recall_ratio']:.2f}"] for c in fr["comparison"]],
           [80, 18, 18, 32, 20, 16], size=7)
    wv = fr.get("with_vs_without")
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 8, f"With vs without fairness - full test set ({fr.get('n_test')} images)", ln=True)
    if wv:
        _table(pdf, ["Metric", "Without fairness", "With fairness (deployed)", "Change"],
               _wv_rows(wv), [62, 40, 48, 30])
        pdf.ln(1)
        _table(pdf, ["Recall"] + list(wv["per_class_recall"].keys()),
               [["Without"] + [_pct(v["without"]) for v in wv["per_class_recall"].values()],
                ["With"] + [_pct(v["with"]) for v in wv["per_class_recall"].values()],
                ["Change"] + [f"{v['change']*100:+.1f}" for v in wv["per_class_recall"].values()]],
               [22] + [24] * len(wv["per_class_recall"]), size=7)
    else:
        pdf.set_font("helvetica", "", 9)
        _para(pdf, 5, "Not available: train the baseline with train_nofair.bat (it re-runs fairness.py), then restart the app.")
    for att, gm in (fr.get("demographic") or {}).items():
        pdf.ln(2)
        pdf.set_font("helvetica", "", 9)
        _para(pdf, 5, _ascii(f"[{att}] accuracy gap {gm['accuracy_gap']*100:.1f} | macro-F1 gap "
                                    f"{gm['macro_f1_gap']*100:.1f} | equalized-odds gap {gm['equalized_odds_gap']*100:.1f} points"))
    if os.path.exists(FAIRNESS_PNG):
        pdf.add_page()
        pdf.image(FAIRNESS_PNG, x=10, w=190)


def _wv_rows(wv):
    rows = []
    for k, m in wv["metrics"].items():
        if k == "recall_ratio":
            rows.append([m["label"], f"{m['without']:.3f}", f"{m['with']:.3f}", f"{m['change']:+.3f}"])
        elif k == "recall_gap":
            rows.append([m["label"], f"{m['without']*100:.1f} pts", f"{m['with']*100:.1f} pts", f"{m['change']*100:+.1f} pts"])
        else:
            rows.append([m["label"], _pct(m["without"]), _pct(m["with"]), f"{m['change']*100:+.1f} pts"])
    ff = wv.get("four_fifths", {})
    rows.append(["Passes four-fifths rule", "yes" if ff.get("without") else "no", "yes" if ff.get("with") else "no", ""])
    return rows


def _with_without_section(pdf, rows, y, p, idx):
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, "2. With vs without fairness (same images)", ln=True)
    pdf.set_font("helvetica", "", 9)
    if not rows or any(r.get("base_pred") is None for r in rows):
        _para(pdf, 5, "Baseline without fairness not loaded. Run train_nofair.bat once (about 40 min on the GPU): "
                      "it trains the same ResNet-50 + VGGFace2 model with no class weights and accuracy-based model "
                      "selection, then re-runs fairness.py. Restart the app and generate the report again.")
        pdf.ln(2)
        return
    _para(pdf, 5, _ascii("Without fairness = same network trained with no class weights, best epoch chosen by accuracy, "
                         "no calibration. With fairness = deployed model (class-weighted loss, macro-F1 selection, "
                         "ensemble + per-class calibration). Both see exactly the same images."))
    pdf.ln(1)
    b = [idx.get(r["base_pred"], -1) for r in rows]
    names = EMOTION_NAMES
    def system(pred):
        pcm = per_class_metrics(y, pred, names)
        return {"per_class": pcm, "summary": disparity_summary(pcm, y, pred)}
    wv = compare_with_without(system(b), system(p))
    _table(pdf, ["Metric", "Without fairness", "With fairness", "Change"], _wv_rows(wv), [62, 40, 48, 30])
    pdf.ln(1)
    _table(pdf, ["Recall"] + list(wv["per_class_recall"].keys()),
           [["Without"] + [_pct(v["without"]) for v in wv["per_class_recall"].values()],
            ["With"] + [_pct(v["with"]) for v in wv["per_class_recall"].values()]],
           [22] + [24] * len(wv["per_class_recall"]), size=7)
    fixed = sum(1 for r in rows if r["correct"] and r["base_pred"] != r["true"])
    broken = sum(1 for r in rows if not r["correct"] and r["base_pred"] == r["true"])
    pdf.ln(1)
    _para(pdf, 5, f"Fairness changed {sum(1 for r in rows if r['base_pred'] != r['pred'])} of {len(rows)} predictions: "
                  f"{fixed} corrected, {broken} made wrong.")
    pdf.ln(2)
