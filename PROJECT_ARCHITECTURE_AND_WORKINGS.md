# System Architecture and Workings

This document explains how FA&EFER is built and how a prediction is produced. For setup and results see the [README](README.md).

## 1. Goal

Most facial emotion recognition (FER) models report one overall accuracy, which hides weak minority classes (RAF-DB: Happiness 4,772 training images vs Fear 281), start from ImageNet object features, and give no reason for a decision. FA&EFER addresses this with:

1. **Face-domain features** — backbones initialised with VGGFace2 face-recognition weights.
2. **Imbalance-aware, honest evaluation** — class-weighted losses, model selection on a validation split by macro-F1, per-class reporting, a test set used once.
3. **Explainability** — Grad-CAM heatmaps and facial Action Units (FACS) for every prediction.

## 2. Components

| Layer | File | Role |
|---|---|---|
| Web UI | `static/index.html` | Upload / webcam / sample images; shows emotion, heatmap and the top Action Units; triggers the batch report |
| API | `app.py` (FastAPI + Uvicorn) | Loads models once at start-up; serves `/predict`, `/generate-report`, `/api/samples` |
| Emotion engine | `train_emotion_resnet50_vggface2.py`, `ensemble_emotion.py` | ResNet-50 + VGGFace2 classifiers (7 emotions), combined as a validation-selected ensemble |
| AU engine | `train_rafau.py` | InceptionResnetV1 + VGGFace2 multi-label detector (76 AU labels) |
| FACS rules | `classify_emotion_from_aus()` in `app.py` | Maps AU probabilities to an emotion (explanation and fallback) |
| Explanations | `pytorch-grad-cam` (LayerCAM) | Emotion heatmap on ResNet-50 `layer3`+`layer4`, averaged over all ensemble models and the mirrored image; top-AU heatmap on InceptionResnetV1 `repeat_2`+`mixed_7a`+`block8`. Both smoothed (Gaussian), robust-normalised (30th–99.5th percentile) and faded at the image border |
| Fairness | `fairness.py` | Per-class / per-group fairness metrics, test-set audit, per-prediction reliability context |
| Reporting | `fpdf2` | Batch PDF audit report |

## 3. Emotion engine

### 3.1 Backbone
The official VGGFace2 ResNet-50 (`resnet50_ft`, trained on MS1M then fine-tuned on VGGFace2: 3.31M images, 9,131 identities) converted from Caffe. Its layout differs from torchvision's ResNet-50:

- the stride-2 is on the **first 1×1 convolution** of each down-sampling bottleneck (torchvision puts it on the 3×3);
- the max-pool uses `padding=0, ceil_mode=True`.

`VGGFace2ResNet50` reproduces this layout exactly (output matched the reference implementation to ~1e-7 relative error), replaces the 8,631-way identity layer with `Dropout(0.3) → Linear(2048, 7)`, and `load_vggface2_weights()` aborts unless all 265 backbone tensors load.

### 3.2 Input format
`ToVGGFace2Tensor`: RGB → **BGR**, pixel values **0–255**, minus mean **[91.4953, 103.8827, 131.0912]**, no division by a standard deviation. Images are resized to 224×224.

### 3.3 Training recipe
| Setting | Value |
|---|---|
| Split | Stratified 10% validation split of RAF-DB train (`--split_seed 42`, shared by all models); test used once |
| Augmentation | Random resized crop (scale 0.8–1.0), horizontal flip, rotation ±10° (p = 0.5), colour jitter 0.2, random erasing (p = 0.25) |
| Schedule | 2 head-only epochs, then all layers; 1-epoch linear warm-up + cosine decay; up to 25 epochs; early stopping on validation macro-F1 |
| Optimiser | AdamW, weight decay 1e-4; backbone LR 1e-4, head LR 1e-3; gradient clipping at 5 |
| Loss | Cross-entropy, label smoothing 0.1, class weights `(N / 7·n_c)^p` normalised to mean 1 (p = 0.5 for model 1, 0.3 for models 2–3) |
| Efficiency | Mixed precision (AMP), batch 64; ~90–110 s per epoch on an RTX 4060 laptop GPU |
| Robustness | Per-epoch `*_last.pth` for `--resume`; skips a batch on a transient cuDNN or out-of-memory error instead of crashing |

### 3.4 Ensemble and calibration (`ensemble_emotion.py`)
1. Rebuild the shared validation split.
2. For each model, average the softmax of the image and its horizontal flip.
3. Evaluate every subset of models on validation; keep the subset with the best macro-F1.
4. Search one additive log-prior offset per class on validation (kept only if it improves validation macro-F1 by ≥ 0.3 points).
5. Report test accuracy and macro-F1 once; write `models/emotion_ensemble.json` (`models`, `class_bias`, `classes`, scores).

## 4. Action Unit engine
- **Labels:** each RAF-AU line `<id>.jpg 1+2+5+26` becomes a 76-length multi-hot vector. The 76 label names (including side variants such as L12, R12) are stored in `models/au_labels.json` so the app does not need the dataset.
- **Imbalance:** `pos_weight_c = (N − P_c) / P_c` in `BCEWithLogitsLoss` (from 0.62 for AU25, present in 61.5% of faces, up to 3,680 for single-example labels).
- **Training:** InceptionResnetV1 (VGGFace2), AdamW (lr 2e-4, wd 1e-3), ReduceLROnPlateau on validation macro-F1, 80/20 split.
- **Validation (921 faces):** micro-F1 53.83%, macro-F1 17.35%, Hamming loss 6.35%, exact match 4.02%. The macro score is low because many labels are rare side variants.

## 5. FACS rule engine
1. `canonical_au()` maps label variants (L12, R12, B12, T23, AD19, …) to the base AU number, keeping the highest probability.
2. Each emotion accumulates weighted evidence from its key AUs and is suppressed by contradicting AUs, e.g.

| Emotion | Key AUs |
|---|---|
| Happiness | AU6 + AU12 (Duchenne smile) |
| Surprise | AU1 + AU2 + AU5 + AU26/27 |
| Fear | AU1 + AU2 + AU20 (+ AU27) |
| Sadness | AU1 + AU4 + AU15/17 |
| Disgust | AU9 + AU10 |
| Anger | AU4 + AU23/24 |

3. The highest score wins; below 0.4 the result is Neutral / Unknown.

On RAF-DB test images the rules alone reach about 31% accuracy (175 images), so they explain the decision rather than make it.

## 6. Request flow (`POST /predict`)
1. FastAPI receives the upload; Pillow decodes it to RGB (unreadable files return HTTP 500 with a message).
2. **AU engine:** ImageNet-normalised 224×224 tensor → 76 sigmoid probabilities; AUs above 0.35 are ranked and the top 5 are returned with anatomical descriptions; LayerCAM over three depths for the top AU (`grad_cam`; flip-averaged only for symmetric AUs, since L/R AUs swap sides when mirrored).
3. **Emotion engine:** VGGFace2-format tensor → each ensemble model on the image and its flip → averaged softmax → per-class offsets → emotion and confidence; LayerCAM on `layer3`+`layer4` of every ensemble model, on the image and its flip, averaged (`emotion_grad_cam`). The UI shows this heatmap by default, with a toggle to the AU heatmap.
4. If no ResNet-50 emotion model is present the app falls back to the older InceptionResnetV1 checkpoint, and finally to the FACS rules.
5. The JSON response also carries the emotion metadata (colour, focal AUs, FACS rule) used by the UI.

## 7. Batch PDF report (`POST /generate-report`)
Samples 5 random images from each test folder, runs the same inference path, and writes the image, AU heatmap, predicted vs true label and detected AUs per row, plus a summary page. Because it samples classes equally and uses only 35 images, its accuracy differs from the test-set accuracy by several points between runs.

## 8. Fairness metrics (`fairness.py`)
Without demographic labels, each emotion class is treated as a group (equal-opportunity view).

| Metric | Meaning |
|---|---|
| Recall (TPR) per class, FNR, FPR | How often each true emotion is recognised / missed / wrongly predicted |
| Recall gap, recall ratio | Best minus worst class recall; worst ÷ best (≥ 0.80 passes the four-fifths rule) |
| Balanced accuracy vs accuracy | A large difference means accuracy is carried by majority classes |
| Prediction-share bias | Share of predictions minus true share per class (over/under-prediction) |
| Minority recall gap | Mean recall of classes with < 7% of training images vs the rest |
| Group accuracy / macro-F1 gap, equalized-odds gap | Only with demographic attributes (`--attributes`) |

**Audit:** `python fairness.py --data_dir TestDb/DATASET` evaluates each single model, the uncalibrated ensemble and the deployed calibrated ensemble on the full test set (a mitigation comparison) and writes `models/fairness_report.json` / `.png`.

**With vs without fairness:** the baseline `models/emotion_r50_nofair.pth` (`train_nofair.bat`) uses the same network, data and seed as model 1, but no class weights, best epoch by validation accuracy and no calibration. `fairness.py` compares it with the deployed system (`with_vs_without` in the JSON); the app runs both on every image, so `/predict` returns `without_fairness` and the batch PDF shows the comparison on the sampled images and on the full test set.

**Per prediction** (`prediction_context`): confidence, top-2 margin, normalised entropy, the class's test recall/precision and training share, FACS rule agreement → reliability band (Low if confidence < 0.5 or class F1 < 0.78; High if confidence ≥ 0.7 and class F1 ≥ 0.85) and a review flag (Low band or margin < 0.15). Returned by `/predict` as `fairness` and summarised on the batch report's Fairness & Reliability page.

## 9. Evaluation integrity
Earlier versions of this project reported 91.4–94.3% from the batch report. That figure was not a measurement: the report passed the ground-truth label into the prediction function and inserted a fixed number of random mismatches, and `/predict` returned the stored label for any test-set filename. Both were removed in v2.0; all predictions now come from the model, and accuracy is measured once on the full 3,068-image test set.

## 10. Known limitations
- No face detection/alignment in the app yet; accuracy on webcam images is lower than on aligned test faces.
- Fear and Disgust remain the weakest classes.
- No demographic labels, so fairness is analysed per class (the demographic audit needs RAF-DB's manual attribute files).
- The AU engine still uses the InceptionResnetV1 backbone.
