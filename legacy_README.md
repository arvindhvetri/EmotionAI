# Legacy experiments

Kept for reference and reproducibility of earlier reviews. They are **not** used by the current app.

| File | What it was |
|---|---|
| `notebook_code.py` | Review 1 baseline: ResNet-18 (ImageNet fallback), weighted cross-entropy, stratified 15% validation split, fairness audit, Grad-CAM (83.67% test accuracy) |
| `train_emotion_model.py` | First InceptionResnetV1 + VGGFace2 emotion model (plain cross-entropy, selected on the test set) |
| `train_balanced_emotion.py` | InceptionResnetV1 + VGGFace2 emotion model with class weights (N/7n)^0.6 and label smoothing |
| `fine_tune_more.py` | Extra fine-tuning of the InceptionResnetV1 emotion model (~81.6% test accuracy, selected on the test set) |
| `train_resnet50.py` | ResNet-50 with **ImageNet** weights + WeightedRandomSampler (comparison only) |
| `build_rafau_notebook.py` | Generator for `notebooks/FA_EFER_ResNet18_VGGFace2.ipynb` (AU EDA, AU metrics, FACS evaluation, Grad-CAM) |

Paths inside these scripts point to the original `D:\Research4` layout and may need adjusting.
