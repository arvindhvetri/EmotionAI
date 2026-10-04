# Emotion AI: Explainable Facial Action Unit & Emotion Recognition Studio

An explainable facial expression recognition and Facial Action Unit (FAU) analysis system powered by **VGGFace2 InceptionResnetV1**, **Paul Ekman's Facial Action Coding System (FACS)** ground truth rules, and **Grad-CAM visual attention heatmaps**.

---

## Key Features

- **76 Facial Action Units (FAU) Detection**: Continuous probability extraction across 76 individual facial muscle deformations using an InceptionResnetV1 architecture trained on RAF-AU.
- **FACS-Based Psychological Classification**: Derives the 7 primary emotions (*Happiness, Surprise, Fear, Sadness, Disgust, Anger, Contempt*) through canonical Ekman FACS combination rules (including Duchenne smile validation, asymmetrical smirk identification, and reciprocal emotional suppression).
- **Grad-CAM Visual Explainability**: Real-time convolutional activation mapping on InceptionResnetV1 `block8` showing exact facial regions that drove the Action Unit predictions.
- **Modern 2-Panel UI Studio**:
  - **Left Panel (Input)**: Drag & drop upload, instant test samples from RAF-DB TestDb, live webcam capture, file metadata, and batch PDF benchmark generation.
  - **Right Panel (Output)**: Emotion hero banner, interactive **Split Slider** (before/after curtain reveal), Side-by-Side comparison, Overlay Blend slider, and ranked Action Units breakdown with confidence progress meters.
- **Batch Evaluation & PDF Report**: Automated evaluation across 35 multi-class TestDb benchmark samples with per-emotion accuracy breakdowns.

---

## Project Structure

```text
├── app.py                     # FastAPI backend & FACS inference pipeline
├── RAFAU_label.txt            # 76 Action Unit class labels definition
├── requirements.txt           # Python library dependencies
├── run.bat                    # One-click Windows startup script
├── run.ps1                    # PowerShell startup script
├── .gitignore                 # Git ignore configuration
├── static/
│   └── index.html             # 2-Panel Explainability Studio frontend
├── models/                    # Trained neural network weights (~90 MB each)
│   ├── best_vggface2_raf_au.pth
│   ├── best_vggface2_emotion.pth
│   └── best_resnet50_raf_au.pth
└── TestDb/                    # Benchmark test dataset & ground truth labels
    ├── DATASET/test/          # Evaluation images across emotion classes 1-7
    ├── test_labels.csv
    └── train_labels.csv
```

---

## Quickstart

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd FA&EFER-App
```

### 2. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Studio
Using the startup script:
```bash
# Windows Batch
run.bat

# Or directly with Uvicorn
uvicorn app:app --reload --port 8000
```
Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## Ekman FACS Emotion Mapping Rules

| Emotion | Defining Action Units | Psychological Ground Truth |
| :--- | :--- | :--- |
| **Happiness** | AU12 + AU6 (± AU25) | Zygomaticus major lip puller paired with Orbicularis oculi cheek raiser (Duchenne Marker). |
| **Surprise** | AU1+2 + AU5 + AU26/27 | High arched brows, widened palpebral fissure, relaxed mandibular drop. |
| **Fear** | AU20 + AU1+2 + AU27/16 | Lateral lip stretcher with terror mouth opening and arched brows. |
| **Sadness** | AU1+4 + AU15/17 | Medial brow knot (corrugator + frontalis) and downward commissure depressors. |
| **Disgust** | AU9 + AU10 (± AU19) | Nasal wrinkling and upper labial levator elevation. |
| **Anger** | AU4 + AU23/24 (± AU16) | Lowered brow furrow with tightened, pressed, or bared dental margins. |
| **Contempt** | L12/R12 or AU14 | Unilateral smirk or unilateral dimple (facial asymmetry marker). |

---

## Git & Large File Management

The model checkpoint files in `models/` are approximately 90 MB each:
- `models/best_vggface2_raf_au.pth` (~90 MB)
- `models/best_vggface2_emotion.pth` (~90 MB)
- `models/best_resnet50_raf_au.pth` (~90 MB)

### Pushing with Git LFS (Recommended)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

---

## License

Academic and research use only.
