# Changelog

## [2.1.0] – 2026-10-26

### Fairness metrics
- New `fairness.py`: per-class recall/precision/F1/FNR/FPR, recall gap and ratio (four-fifths rule), balanced accuracy vs accuracy, prediction-share bias, minority-vs-majority recall gap; per-group metrics and equalized-odds gap for demographic attributes (optional `--attributes`).
- CLI audit on the full RAF-DB test set compares single models, the uncalibrated ensemble and the deployed calibrated ensemble → `models/fairness_report.json` + `.png`.
- `/predict` returns `emotion_probabilities` and a `fairness` object (confidence, margin, uncertainty, class test recall/precision, minority flag, FACS agreement, reliability band, review flag, model audit summary); the UI shows it in a "Fairness & Reliability" card.
- Batch PDF: reliability per row and a "Fairness & Reliability" page (sample metrics, review flags, FACS agreement, full test-set audit and mitigation comparison).
- `ensemble_emotion.py` prints a one-line fairness summary.
- **Heatmaps:** Grad-CAM replaced by LayerCAM on several layers (finer than the 7×7 / 5×5 last layer), averaged over ensemble models and the mirrored image, smoothed and robust-normalised. The UI now shows the *emotion* heatmap by default (previously only the AU heatmap) with a toggle; the PDF shows both per image. FACS cross-check limited to the 7 RAF-DB emotions (no Contempt).
- **With vs without fairness:** `--select_metric` option in training; `train_nofair.bat` trains a baseline without fairness mitigation (`models/emotion_r50_nofair.pth`); `fairness.py` reports both systems side by side; the app runs the baseline alongside the deployed model (`without_fairness` in `/predict`, a row per image and a comparison section in the batch PDF).

## [2.0.0] – 2026-10-21

### Emotion engine: ResNet-50 + VGGFace2
- New `train_emotion_resnet50_vggface2.py`: official VGGFace2 ResNet-50 (`resnet50_ft`) in its original Caffe layout, VGGFace2 input format (BGR, 0–255, mean subtraction), hard failure if weights do not load.
- Two-phase fine-tuning (head warm-up, then all layers), discriminative learning rates, cosine schedule, label smoothing, softened class weights, augmentation, mixed precision, early stopping on validation macro-F1.
- Crash-safe training: per-epoch `*_last.pth`, `--resume`, `--eval_only`, recovery from transient cuDNN / out-of-memory errors.
- New `ensemble_emotion.py`: validation-selected ensemble with optional per-class calibration → `models/emotion_ensemble.json`.
- Results on the RAF-DB test set (3,068 images): single models 89.15% / 90.22% accuracy; ensemble **90.38% accuracy, 85.65% macro-F1** (previous checkpoint: about 81.6%, selected on the test set).

### Evaluation fixes
- Removed the ground-truth override and injected mismatches from the batch PDF report, and the test-set filename lookup from `/predict`. All predictions now come from the model.
- Model selection uses a stratified validation split; the test set is used once.
- RAF-DB class 7 renamed from "Contempt" to **Neutral** (app, UI, reports).

### App
- Loads the ensemble (or a single ResNet-50 model, falling back to the old model); flip test-time augmentation; per-class calibration.
- `/predict` adds `emotion_confidence`, `emotion_grad_cam` (Grad-CAM for the emotion decision) and `emotion_model`.
- AU label names shipped in `models/au_labels.json`, so the app runs without the RAF-AU dataset.

### Tooling and docs
- `train_emotion.bat`, `train_more.bat` (Windows, always use the project venv).
- Google Colab notebook `notebooks/FA_EFER_ResNet50_VGGFace2_Colab.ipynb`.
- Rewritten README, architecture document, `DATASETS.md`; pinned `requirements.txt`.

### Repository clean-up
- Datasets (RAF-DB, RAF-AU annotations) removed from version control — their licence forbids redistribution.
- Scratch/debug scripts and intermediate files removed; earlier experiments moved to `legacy/`; notebooks moved to `notebooks/`.

## [1.0.0] – 2026-10-04
- Dual-engine prototype: InceptionResnetV1 + VGGFace2 AU detector (RAF-AU, 76 AUs) and emotion classifier (RAF-DB); FACS rule engine; Grad-CAM on `block8`; FastAPI two-panel UI; batch PDF report.
- Research notebook with AU EDA, multi-label metrics (validation micro-F1 53.83%, macro-F1 17.35%), FACS cross-dataset evaluation (31% on 175 RAF-DB test images) and Grad-CAM.
- **Correction:** the 91.4–94.3% "batch accuracy" reported for this version was not a measurement (see v2.0 evaluation fixes).

### Earlier iterations
- Hand-written FACS rules with binary AU matching: roughly 35–40% on batch reports.
- `canonical_au()` normalisation of L/R/B/T AU variants.
- Weighted continuous AU scoring: 42.9% on a random 35-image batch.
- ResNet-18 + ImageNet baseline (Review 1): 83.67% test accuracy, Disgust F1 0.5686.
