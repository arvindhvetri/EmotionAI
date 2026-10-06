# ============================================================
# 2a. Install any missing packages (uncomment as needed)
# ============================================================
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install numpy pandas scikit-learn matplotlib seaborn tqdm Pillow
# ============================================================
# 2b. Imports
# ============================================================
import os
import sys
import copy
import json
import random
import warnings
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision
from torchvision import transforms, models

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
%matplotlib inline

print("=" * 60)
print("ENVIRONMENT INFORMATION")
print("=" * 60)
print(f"Python      : {sys.version}")
print(f"PyTorch     : {torch.__version__}")
print(f"Torchvision : {torchvision.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
import sklearn; print(f"Scikit-learn : {sklearn.__version__}")
print("=" * 60)
# ============================================================
# 2c. CONFIGURATION - Edit these values as needed
# ============================================================

# --- Reproducibility ---
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n>> Selected device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# --- Hyperparameters ---
BATCH_SIZE = 32           # Lower to 16 for CPU-only / low-VRAM
NUM_EPOCHS = 25           # Adjust: 20-30 epochs is a good starting range
LEARNING_RATE = 1e-4      # For AdamW fine-tuning
WEIGHT_DECAY = 1e-4
PATIENCE = 7              # Early-stopping patience (epochs)
VAL_SPLIT = 0.15          # Fraction of training data to use as validation
NUM_WORKERS = 2           # DataLoader workers (set to 0 on Windows if issues)
NUM_CLASSES = 7

# --- Image size ---
# Placeholder - will be FINALISED after inspecting actual dataset dimensions
# in Section 3 (Data Understanding). ResNet-18 default is 224x224.
IMG_SIZE = 224  # Will be justified in Data Understanding

# --- Dataset paths ---
DATASET_ROOT    = r"Dataset"                              # Relative to notebook
TRAIN_IMG_DIR   = os.path.join(DATASET_ROOT, "Images", "train")
TEST_IMG_DIR    = os.path.join(DATASET_ROOT, "Images", "test")
TRAIN_CSV_PATH  = os.path.join(DATASET_ROOT, "train_labels.csv")
TEST_CSV_PATH   = os.path.join(DATASET_ROOT, "test_labels.csv")

# --- VGGFace2 Checkpoint Path (IMPORTANT - read Checkpoint Fallback Policy) ---
# Place a VGGFace2-pretrained ResNet-18 .pth checkpoint here if available.
# If this file does not exist, the notebook will FALL BACK to ImageNet weights
# with a clearly visible warning.
VGGFACE2_RESNET18_CHECKPOINT_PATH = r""  # e.g., r"checkpoints/resnet18_vggface2.pth"

# --- Model save path ---
MODEL_SAVE_DIR  = "models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_resnet18_vggface2_rafdb.pth")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("\n>> Configuration set successfully.")
print(f"   Batch size  : {BATCH_SIZE}")
print(f"   Epochs      : {NUM_EPOCHS}")
print(f"   LR          : {LEARNING_RATE}")
print(f"   Val split   : {VAL_SPLIT}")
print(f"   Image size  : {IMG_SIZE}")
print(f"   Num classes : {NUM_CLASSES}")
# ============================================================
# 3a. Load and inspect CSV files
# ============================================================
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

print("=" * 60)
print("TRAIN CSV - First 5 Rows")
print("=" * 60)
display(train_df.head())

print("\nTEST CSV - First 5 Rows")
print("=" * 60)
display(test_df.head())

print(f"\nTrain CSV shape : {train_df.shape}")
print(f"Test  CSV shape : {test_df.shape}")

print(f"\nTrain columns   : {list(train_df.columns)}")
print(f"Test  columns   : {list(test_df.columns)}")

print("\nTrain dtypes:")
print(train_df.dtypes)
print("\nTest dtypes:")
print(test_df.dtypes)
# ============================================================
# 3b. Missing values, duplicates, unique label values
# ============================================================
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print("Train:")
print(train_df.isnull().sum())
print("\nTest:")
print(test_df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(f"Train duplicates: {train_df.duplicated().sum()}")
print(f"Test  duplicates: {test_df.duplicated().sum()}")

print("\n" + "=" * 60)
print("UNIQUE LABEL VALUES")
print("=" * 60)
print(f"Train labels: {sorted(train_df['label'].unique())}")
print(f"Test  labels: {sorted(test_df['label'].unique())}")
# ============================================================
# 3c. Identify demographic columns
# ============================================================
POSSIBLE_DEMOGRAPHIC_COLS = ["gender", "age", "age_group", "race", "ethnicity", "sex"]

demographic_cols_train = [c for c in train_df.columns if c.lower() in POSSIBLE_DEMOGRAPHIC_COLS]
demographic_cols_test  = [c for c in test_df.columns  if c.lower() in POSSIBLE_DEMOGRAPHIC_COLS]

HAS_DEMOGRAPHICS = len(demographic_cols_test) > 0

if HAS_DEMOGRAPHICS:
    print(f"\nDemographic columns FOUND in train CSV: {demographic_cols_train}")
    print(f"Demographic columns FOUND in test CSV : {demographic_cols_test}")
else:
    print("\n[WARNING] No demographic columns (gender, age, race, etc.) found in the CSV files.")
    print("Full demographic fairness analysis cannot be performed directly.")
    print("The Fairness Analysis section will note this limitation and provide")
    print("a ready-to-use framework for when demographic annotations are added.")
# ============================================================
# 3d. Image counts per class folder
# ============================================================

def count_images_per_folder(root_dir):
    """Count image files in each class sub-folder."""
    counts = {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            n = sum(1 for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in valid_ext)
            counts[folder_name] = n
    return counts

train_folder_counts = count_images_per_folder(TRAIN_IMG_DIR)
test_folder_counts  = count_images_per_folder(TEST_IMG_DIR)

print("=" * 60)
print("IMAGE COUNTS PER CLASS FOLDER")
print("=" * 60)
print("\nTRAIN folders:")
for k, v in train_folder_counts.items():
    print(f"  Folder {k}: {v} images")
print(f"  TOTAL: {sum(train_folder_counts.values())} images")

print("\nTEST folders:")
for k, v in test_folder_counts.items():
    print(f"  Folder {k}: {v} images")
print(f"  TOTAL: {sum(test_folder_counts.values())} images")
# ============================================================
# 3e. Verify CSV-to-folder consistency
# ============================================================
csv_train_count = len(train_df)
folder_train_count = sum(train_folder_counts.values())
csv_test_count = len(test_df)
folder_test_count = sum(test_folder_counts.values())

print("\n" + "=" * 60)
print("CSV vs FOLDER CONSISTENCY CHECK")
print("=" * 60)
print(f"Train - CSV rows: {csv_train_count}, Folder images: {folder_train_count}")
print(f"Test  - CSV rows: {csv_test_count},  Folder images: {folder_test_count}")

if csv_train_count == folder_train_count:
    print("[OK] Train counts match.")
else:
    print(f"[WARNING] Train count MISMATCH: CSV={csv_train_count}, Folders={folder_train_count}")

if csv_test_count == folder_test_count:
    print("[OK] Test counts match.")
else:
    print(f"[WARNING] Test count MISMATCH: CSV={csv_test_count}, Folders={folder_test_count}")
# ============================================================
# 3f. Detect actual image dimensions
# ============================================================

def sample_image_dimensions(root_dir, samples_per_class=3):
    """Sample a few images per class folder and return their (width, height)."""
    dims = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]
        for f in files[:samples_per_class]:
            img = Image.open(os.path.join(folder_path, f))
            dims.append((img.width, img.height))
    return dims

train_dims = sample_image_dimensions(TRAIN_IMG_DIR, samples_per_class=5)

widths  = [d[0] for d in train_dims]
heights = [d[1] for d in train_dims]

print("\n" + "=" * 60)
print("IMAGE DIMENSION ANALYSIS (sampled from training set)")
print("=" * 60)
print(f"Sampled {len(train_dims)} images")
print(f"Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
print(f"Unique dimensions: {set(train_dims)}")

# Justify resize target
print(f"\n>> Chosen resize target: {IMG_SIZE}x{IMG_SIZE}")
print("   Justification: ResNet-18 is designed for 224x224 input. RAF-DB aligned")
print("   images are typically 100x100, so we UP-SAMPLE to 224x224 to match the")
print("   pretrained architecture's expected receptive field and feature-map sizes.")
# ============================================================
# 3g. Determine and verify emotion label mapping
# ============================================================
# RAF-DB standard 7-class basic emotion mapping (1-indexed):
# 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Contempt
#
# We verify this against the observed labels in the CSV.

observed_labels = sorted(train_df['label'].unique())
print(f"Observed labels in train CSV: {observed_labels}")
print(f"Number of unique labels     : {len(observed_labels)}")

# Define the mapping - adjust here if your dataset uses a different convention
LABEL_TO_EMOTION = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Contempt",
}

EMOTION_TO_LABEL = {v: k for k, v in LABEL_TO_EMOTION.items()}

# Since PyTorch expects 0-indexed class labels, we create a shifted mapping
# Folder label i (1-7) -> model index (i-1) = 0-6
IDX_TO_EMOTION = {i - 1: name for i, name in LABEL_TO_EMOTION.items()}
EMOTION_TO_IDX = {name: i - 1 for i, name in LABEL_TO_EMOTION.items()}
EMOTION_NAMES = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]

print("\nFinal label mapping (folder label -> 0-indexed model index -> emotion):")
for folder_lbl, emotion in LABEL_TO_EMOTION.items():
    print(f"  Folder {folder_lbl}  ->  Index {folder_lbl - 1}  ->  {emotion}")

print(f"\nEmotion names (ordered by model index): {EMOTION_NAMES}")
# ============================================================
# 4a. Class distribution - Training
# ============================================================
train_label_counts = train_df['label'].value_counts().sort_index()
test_label_counts  = test_df['label'].value_counts().sort_index()

print("=" * 60)
print("TRAINING SET - CLASS DISTRIBUTION")
print("=" * 60)
for lbl, cnt in train_label_counts.items():
    pct = cnt / len(train_df) * 100
    emotion = LABEL_TO_EMOTION.get(lbl, f"Unknown({lbl})")
    print(f"  Label {lbl} ({emotion:>10s}): {cnt:>5d} images  ({pct:5.1f}%)")

print(f"\n  Total training images: {len(train_df)}")

print("\n" + "=" * 60)
print("TEST SET - CLASS DISTRIBUTION")
print("=" * 60)
for lbl, cnt in test_label_counts.items():
    pct = cnt / len(test_df) * 100
    emotion = LABEL_TO_EMOTION.get(lbl, f"Unknown({lbl})")
    print(f"  Label {lbl} ({emotion:>10s}): {cnt:>5d} images  ({pct:5.1f}%)")

print(f"\n  Total test images: {len(test_df)}")
# ============================================================
# 4b. Class imbalance discussion
# ============================================================
majority_label = train_label_counts.idxmax()
minority_label = train_label_counts.idxmin()
imbalance_ratio = train_label_counts.max() / train_label_counts.min()

print("\n" + "=" * 60)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 60)
print(f"Majority class: Label {majority_label} ({LABEL_TO_EMOTION[majority_label]}) "
      f"- {train_label_counts.max()} samples")
print(f"Minority class: Label {minority_label} ({LABEL_TO_EMOTION[minority_label]}) "
      f"- {train_label_counts.min()} samples")
print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
print("\nRAF-DB is known to be significantly class-imbalanced. 'Happiness' is")
print("typically the majority class by a large margin, while 'Contempt', 'Fear',")
print("and 'Disgust' tend to be under-represented.")
print("\nImplications for training:")
print("  - Standard cross-entropy will be dominated by majority-class gradients.")
print("  - We will use WEIGHTED cross-entropy loss (inverse frequency weights)")
print("    to mitigate this during training.")
print("  - We use Macro F1 as the primary validation metric instead of accuracy")
print("    to give equal importance to all classes.")
# ============================================================
# 4c. Train vs Test distribution comparison
# ============================================================
comparison_df = pd.DataFrame({
    "Emotion": [LABEL_TO_EMOTION[l] for l in sorted(LABEL_TO_EMOTION.keys())],
    "Train Count": [train_label_counts.get(l, 0) for l in sorted(LABEL_TO_EMOTION.keys())],
    "Test Count":  [test_label_counts.get(l, 0) for l in sorted(LABEL_TO_EMOTION.keys())],
})
comparison_df["Train %"] = (comparison_df["Train Count"] / comparison_df["Train Count"].sum() * 100).round(1)
comparison_df["Test %"]  = (comparison_df["Test Count"]  / comparison_df["Test Count"].sum()  * 100).round(1)

print("\n" + "=" * 60)
print("TRAIN vs TEST DISTRIBUTION COMPARISON")
print("=" * 60)
display(comparison_df)
# ============================================================
# 4d. Demographic analysis (if columns exist)
# ============================================================
if HAS_DEMOGRAPHICS:
    for col in demographic_cols_train:
        print(f"\n--- {col.upper()} DISTRIBUTION (Train) ---")
        print(train_df[col].value_counts())
    for col in demographic_cols_test:
        print(f"\n--- {col.upper()} DISTRIBUTION (Test) ---")
        print(test_df[col].value_counts())
else:
    print("\n[INFO] No demographic columns found. Skipping demographic EDA.")
    print("       Columns present: ", list(train_df.columns))
# ============================================================
# 5a. Bar chart - Emotion class distribution (Train & Test)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

colors_palette = sns.color_palette("mako", n_colors=NUM_CLASSES)

# Train
train_emo_names = [LABEL_TO_EMOTION[l] for l in train_label_counts.index]
axes[0].bar(train_emo_names, train_label_counts.values, color=colors_palette)
axes[0].set_title("Training Set - Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Emotion", fontsize=12)
axes[0].set_ylabel("Number of Images", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)
for i, (name, val) in enumerate(zip(train_emo_names, train_label_counts.values)):
    axes[0].text(i, val + 30, str(val), ha="center", fontsize=9, fontweight="bold")

# Test
test_emo_names = [LABEL_TO_EMOTION[l] for l in test_label_counts.index]
axes[1].bar(test_emo_names, test_label_counts.values, color=colors_palette)
axes[1].set_title("Test Set - Class Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Emotion", fontsize=12)
axes[1].set_ylabel("Number of Images", fontsize=12)
axes[1].tick_params(axis="x", rotation=45)
for i, (name, val) in enumerate(zip(test_emo_names, test_label_counts.values)):
    axes[1].text(i, val + 10, str(val), ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()
# ============================================================
# 5b. Train vs Test comparison (grouped bar)
# ============================================================
x = np.arange(NUM_CLASSES)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
train_pcts = comparison_df["Train %"].values
test_pcts  = comparison_df["Test %"].values

bars1 = ax.bar(x - width/2, train_pcts, width, label="Train %", color="#4C72B0")
bars2 = ax.bar(x + width/2, test_pcts,  width, label="Test %",  color="#DD8452")

ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Train vs Test - Class Distribution Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(EMOTION_NAMES, rotation=45)
ax.legend(fontsize=11)
ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)

plt.tight_layout()
plt.show()
# ============================================================
# 5c. Sample images from each emotion class (grid)
# ============================================================

def load_sample_images(root_dir, folder_label, n=4):
    """Load n sample images from a given class folder."""
    folder_path = os.path.join(root_dir, str(folder_label))
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in valid_ext]
    files = files[:n]
    images = []
    for f in files:
        img = Image.open(os.path.join(folder_path, f)).convert("RGB")
        images.append(img)
    return images

fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(12, 3 * NUM_CLASSES))

for row_idx, (folder_lbl, emotion) in enumerate(LABEL_TO_EMOTION.items()):
    sample_imgs = load_sample_images(TRAIN_IMG_DIR, folder_lbl, n=4)
    for col_idx in range(4):
        ax = axes[row_idx][col_idx]
        if col_idx < len(sample_imgs):
            ax.imshow(sample_imgs[col_idx])
        ax.axis("off")
        if col_idx == 0:
            ax.set_title(f"{emotion}", fontsize=12, fontweight="bold", loc="left")

plt.suptitle("Sample Images from Each Emotion Class (Training Set)",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
# ============================================================
# 5d. Demographic visualizations (only if data exists)
# ============================================================
if HAS_DEMOGRAPHICS:
    for col in demographic_cols_train:
        fig, ax = plt.subplots(figsize=(8, 4))
        train_df[col].value_counts().plot(kind="bar", ax=ax, color=colors_palette)
        ax.set_title(f"{col.title()} Distribution (Train)", fontsize=14, fontweight="bold")
        ax.set_xlabel(col.title(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.show()

    # Emotion distribution across demographic groups
    for col in demographic_cols_train:
        fig, ax = plt.subplots(figsize=(12, 5))
        ct = pd.crosstab(train_df[col],
                         train_df['label'].map(LABEL_TO_EMOTION),
                         normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title(f"Emotion Distribution across {col.title()} Groups (Train)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
else:
    print("[INFO] No demographic data available - skipping demographic visualizations.")
# ============================================================
# 6a. Define transforms
# ============================================================
# Normalization: ImageNet mean/std are used as the default.
# VGGFace2-pretrained checkpoints from the official repo also typically
# use ImageNet normalization. If your specific checkpoint uses different
# stats, update these values accordingly.

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

print("[OK] Transforms defined.")
print(f"   Train augmentations: HFlip, Rotation(+/-10 deg), ColorJitter, Affine")
print(f"   Normalization: mean={NORMALIZE_MEAN}, std={NORMALIZE_STD}")
# ============================================================
# 6b. Custom Dataset class
# ============================================================

class RAFDBDataset(Dataset):
    """Custom PyTorch Dataset for RAF-DB.
    
    Loads images from class-folder structure (folder names are 1-indexed labels).
    Returns 0-indexed labels suitable for PyTorch's CrossEntropyLoss.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # List of (image_path, 0-indexed_label)
        self.targets = []   # 0-indexed labels for stratified splitting
        
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            folder_label = int(folder_name)       # 1-indexed
            class_idx = folder_label - 1          # 0-indexed
            
            for fname in sorted(os.listdir(folder_path)):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    img_path = os.path.join(folder_path, fname)
                    self.samples.append((img_path, class_idx))
                    self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Return path for later analysis

print("[OK] RAFDBDataset class defined.")
# ============================================================
# 6c. Create datasets and train/val split
# ============================================================

# Full training dataset (before splitting)
full_train_dataset = RAFDBDataset(TRAIN_IMG_DIR, transform=None)  # transform applied later
test_dataset       = RAFDBDataset(TEST_IMG_DIR,  transform=val_test_transform)

# Stratified train/val split
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
train_indices, val_indices = next(sss.split(
    np.zeros(len(full_train_dataset)), full_train_dataset.targets
))

print(f"Full training set   : {len(full_train_dataset)} samples")
print(f"After split - Train : {len(train_indices)} samples")
print(f"After split - Val   : {len(val_indices)} samples")
print(f"Test set            : {len(test_dataset)} samples")
# ============================================================
# 6d. Wrap subsets with correct transforms
# ============================================================

class SubsetWithTransform(Dataset):
    """Wraps a Subset and applies a specific transform."""
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[real_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

train_dataset = SubsetWithTransform(full_train_dataset, train_indices, train_transform)
val_dataset   = SubsetWithTransform(full_train_dataset, val_indices, val_test_transform)

print(f"\nTrain dataset (with augmentation)  : {len(train_dataset)} samples")
print(f"Val   dataset (no augmentation)    : {len(val_dataset)} samples")
print(f"Test  dataset (no augmentation)    : {len(test_dataset)} samples")
# ============================================================
# 6e. Create DataLoaders
# ============================================================
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"\nDataLoaders created:")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val   batches : {len(val_loader)}")
print(f"  Test  batches : {len(test_loader)}")
# ============================================================
# 6f. Visualise one batch of transformed training images
# ============================================================

def denormalize(tensor, mean, std):
    """Reverse normalisation for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# Get one batch
batch_imgs, batch_labels, _ = next(iter(train_loader))

fig, axes = plt.subplots(2, 8, figsize=(20, 5))
for i, ax in enumerate(axes.flat):
    if i >= len(batch_imgs):
        ax.axis("off")
        continue
    img = denormalize(batch_imgs[i], NORMALIZE_MEAN, NORMALIZE_STD)
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(img)
    ax.set_title(IDX_TO_EMOTION[batch_labels[i].item()], fontsize=9)
    ax.axis("off")

plt.suptitle("Sample Batch - Transformed Training Images", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# ============================================================
# 7a. VGGFace2 Checkpoint Loading (with Fallback Policy)
# ============================================================

USING_VGGFACE2 = False  # Will be set to True if checkpoint loads successfully

def build_resnet18_model(num_classes=NUM_CLASSES, checkpoint_path=VGGFACE2_RESNET18_CHECKPOINT_PATH):
    """
    Build a ResNet-18 model for emotion classification.
    
    Priority:
    1. Load VGGFace2-pretrained ResNet-18 weights from checkpoint_path
    2. If unavailable, FALL BACK to ImageNet-pretrained ResNet-18 with explicit warning
    
    Returns:
        model (nn.Module): ResNet-18 with final FC replaced for `num_classes`
        using_vggface2 (bool): Whether VGGFace2 weights were loaded
    """
    global USING_VGGFACE2
    
    # Step 1: Try to load VGGFace2 checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("=" * 70)
        print("LOADING VGGFace2-PRETRAINED ResNet-18 CHECKPOINT")
        print("=" * 70)
        print(f"Checkpoint path: {checkpoint_path}")
        
        # Create a base ResNet-18 (no pretrained weights)
        model = models.resnet18(weights=None)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean up key names (remove 'module.' prefix if saved with DataParallel)
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            cleaned_state_dict[name] = v
        
        # Load with strict=False to allow final-layer shape mismatch
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        
        print(f"\n[OK] VGGFace2 checkpoint loaded SUCCESSFULLY.")
        if result.missing_keys:
            print(f"   Missing keys  : {result.missing_keys}")
        if result.unexpected_keys:
            print(f"   Unexpected keys: {result.unexpected_keys}")
        if not result.missing_keys and not result.unexpected_keys:
            print("   All keys matched perfectly.")
        
        USING_VGGFACE2 = True
    
    else:
        # Step 2: FALLBACK - ImageNet pretrained
        print("=" * 70)
        print("WARNING: VGGFace2 CHECKPOINT NOT FOUND")
        print("=" * 70)
        if checkpoint_path:
            print(f"Specified path: {checkpoint_path}")
            print(f"File exists: {os.path.isfile(checkpoint_path)}")
        else:
            print("No checkpoint path was specified (VGGFACE2_RESNET18_CHECKPOINT_PATH is empty).")
        
        print("\n" + "*" * 70)
        print("* WARNING: VGGFace2 checkpoint not found.                           *")
        print("* Falling back to ImageNet-pretrained ResNet-18.                    *")
        print("* Results are NOT representative of the VGGFace2-based approach     *")
        print("* described in the project.                                         *")
        print("*" * 70)
        print("\nTo use VGGFace2 weights:")
        print("  1. Obtain a VGGFace2-pretrained ResNet-18 checkpoint (.pth file)")
        print("  2. Set VGGFACE2_RESNET18_CHECKPOINT_PATH in Section 2 to its path")
        print("  3. Re-run the notebook from Section 7 onward\n")
        print("Note: Most publicly available VGGFace2 checkpoints are for ResNet-50")
        print("or SE-ResNet-50, not ResNet-18. You may need to convert weights from")
        print("a compatible architecture or obtain from a verified third-party source.\n")
        
        # Load ImageNet-pretrained ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("[OK] ImageNet-pretrained ResNet-18 loaded as FALLBACK.")
        USING_VGGFACE2 = False
    
    # Step 3: Replace final FC layer for 7 emotion classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"\n>> Final FC layer replaced: {in_features} -> {num_classes} classes")
    
    return model, USING_VGGFACE2


# Build the model
model, USING_VGGFACE2 = build_resnet18_model()
model = model.to(DEVICE)

print(f"\n>> Model moved to: {DEVICE}")
print(f">> Weights source : {'VGGFace2' if USING_VGGFACE2 else 'ImageNet (FALLBACK)'}")
# ============================================================
# 7b. Model architecture summary
# ============================================================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 60)
print(model)
print(f"\nTotal parameters     : {total_params:>12,}")
print(f"Trainable parameters : {trainable_params:>12,}")
print(f"Frozen parameters    : {total_params - trainable_params:>12,}")
# ============================================================
# 7c. Loss function, optimizer, and scheduler
# ============================================================

# --- Class weights for imbalanced RAF-DB ---
# Compute inverse-frequency weights from the training subset
train_labels_subset = full_train_dataset.targets[train_indices]
class_counts = np.bincount(train_labels_subset, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # Normalise so they sum to NUM_CLASSES
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

print("Class weights (inverse-frequency, normalised):")
for i, (emo, w) in enumerate(zip(EMOTION_NAMES, class_weights)):
    print(f"  {emo:>10s} (idx {i}): count={class_counts[i]:>5d}, weight={w:.4f}")

# Loss function - weighted cross-entropy to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"\n>> Loss function: Weighted CrossEntropyLoss")
print("   Justification: RAF-DB is significantly class-imbalanced. Weighting ensures")
print("   the loss gives more importance to under-represented classes.")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print(f"\n>> Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

# Learning rate scheduler - reduce on plateau (monitors val Macro F1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)
print(f">> Scheduler: ReduceLROnPlateau (monitors val Macro F1, factor=0.5, patience=3)")
# ============================================================
# 8. Training loop
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns avg loss, accuracy, macro F1."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels, _ in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def validate(model, loader, criterion, device):
    """Validate. Returns avg loss, accuracy, macro F1."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="  Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


# --- Training history ---
history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [],  "val_acc": [],
    "train_f1": [],   "val_f1": [],
}

best_val_f1 = 0.0
best_epoch = 0
epochs_no_improve = 0

print("=" * 70)
print(f"STARTING TRAINING - {NUM_EPOCHS} epochs, Early Stopping patience={PATIENCE}")
print(f"Checkpoint metric: Validation Macro F1")
print(f"Device: {DEVICE}")
print("=" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    
    # Validate
    val_loss, val_acc, val_f1 = validate(
        model, val_loader, criterion, DEVICE
    )
    
    # Record history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    
    # Step scheduler
    scheduler.step(val_f1)
    
    # Print results
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}  |  Train Macro F1: {train_f1:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  |  Val   Macro F1: {val_f1:.4f}")
    
    # Check for improvement
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_no_improve = 0
        
        # Save best checkpoint
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "class_mapping": IDX_TO_EMOTION,
            "config": {
                "img_size": IMG_SIZE,
                "num_classes": NUM_CLASSES,
                "normalize_mean": NORMALIZE_MEAN,
                "normalize_std": NORMALIZE_STD,
                "using_vggface2": USING_VGGFACE2,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        }
        torch.save(checkpoint_data, BEST_MODEL_PATH)
        print(f"  >>> Best model saved! (Val Macro F1: {val_f1:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve}/{PATIENCE} epochs.")
    
    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        break

print("\n" + "=" * 70)
print(f"TRAINING COMPLETE")
print(f"Best epoch: {best_epoch}  |  Best Val Macro F1: {best_val_f1:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")
print("=" * 70)
# ============================================================
# 9. Training visualisation
# ============================================================
epochs_range = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# --- Loss ---
axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#4C72B0")
axes[0].plot(epochs_range, history["val_loss"],   "s-", label="Val Loss",   color="#DD8452")
axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
axes[0].set_title("Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Accuracy ---
axes[1].plot(epochs_range, history["train_acc"], "o-", label="Train Acc", color="#4C72B0")
axes[1].plot(epochs_range, history["val_acc"],   "s-", label="Val Acc",   color="#DD8452")
axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# --- Macro F1 ---
axes[2].plot(epochs_range, history["train_f1"], "o-", label="Train Macro F1", color="#4C72B0")
axes[2].plot(epochs_range, history["val_f1"],   "s-", label="Val Macro F1",   color="#DD8452")
axes[2].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
axes[2].set_title("Macro F1-Score", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Macro F1")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Training & Validation Curves", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
# ============================================================
# 10a. Load best model
# ============================================================
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)

eval_model = models.resnet18(weights=None)
eval_model.fc = nn.Linear(eval_model.fc.in_features, NUM_CLASSES)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model = eval_model.to(DEVICE)
eval_model.eval()

print(f"[OK] Best model loaded from: {BEST_MODEL_PATH}")
print(f"   Trained for epoch : {checkpoint['epoch']}")
print(f"   Val Macro F1      : {checkpoint['val_macro_f1']:.4f}")
print(f"   Val Accuracy      : {checkpoint['val_acc']:.4f}")
print(f"   Weights source    : {'VGGFace2' if checkpoint['config']['using_vggface2'] else 'ImageNet (FALLBACK)'}")
# ============================================================
# 10b. Full test set evaluation
# ============================================================
all_preds = []
all_labels = []
all_probs = []
all_paths = []

eval_model.eval()
running_test_loss = 0.0

with torch.no_grad():
    for images, labels, paths in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        outputs = eval_model(images)
        loss = criterion(outputs, labels)
        running_test_loss += loss.item() * images.size(0)
        
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

test_loss = running_test_loss / len(test_dataset)
test_acc  = accuracy_score(all_labels, all_preds)
test_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
test_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
test_f1_macro    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
test_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"  Test Loss         : {test_loss:.4f}")
print(f"  Accuracy          : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Macro Precision   : {test_prec:.4f}")
print(f"  Macro Recall      : {test_rec:.4f}")
print(f"  Macro F1-Score    : {test_f1_macro:.4f}")
print(f"  Weighted F1-Score : {test_f1_weighted:.4f}")
# ============================================================
# 10c. Classification report
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    all_labels, all_preds,
    target_names=EMOTION_NAMES,
    digits=4,
    zero_division=0
))
# ============================================================
# 10d. Confusion matrices
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
            ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Predicted", fontsize=12)
axes[0].set_ylabel("Actual", fontsize=12)

# Normalised confusion matrix
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
            ax=axes[1])
axes[1].set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Predicted", fontsize=12)
axes[1].set_ylabel("Actual", fontsize=12)

plt.tight_layout()
plt.show()

# Most confused pairs
print("\nMost Frequently Confused Emotion Pairs:")
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)  # Zero out diagonal to find off-diagonal maxima
top_confusions = []
cm_flat = cm_offdiag.copy()
for _ in range(5):
    idx = np.unravel_index(cm_flat.argmax(), cm_flat.shape)
    if cm_flat[idx] == 0:
        break
    top_confusions.append((EMOTION_NAMES[idx[0]], EMOTION_NAMES[idx[1]], cm_flat[idx]))
    cm_flat[idx] = 0

for actual, predicted, count in top_confusions:
    print(f"  {actual:>10s} -> {predicted:<10s} : {count} samples")
# ============================================================
# 10e. Per-class metrics
# ============================================================
per_class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
per_class_rec  = recall_score(all_labels, all_preds, average=None, zero_division=0)
per_class_f1   = f1_score(all_labels, all_preds, average=None, zero_division=0)

metrics_df = pd.DataFrame({
    "Emotion": EMOTION_NAMES,
    "Precision": per_class_prec,
    "Recall": per_class_rec,
    "F1-Score": per_class_f1,
})
metrics_df = metrics_df.round(4)

print("\n" + "=" * 60)
print("PER-CLASS METRICS")
print("=" * 60)
display(metrics_df)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(NUM_CLASSES)
width = 0.25

ax.bar(x - width, per_class_prec, width, label="Precision", color="#4C72B0")
ax.bar(x,         per_class_rec,  width, label="Recall",    color="#55A868")
ax.bar(x + width, per_class_f1,   width, label="F1-Score",  color="#DD8452")

ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Precision, Recall, and F1-Score", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(EMOTION_NAMES, rotation=45)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# 10f. Example correct and incorrect predictions
# ============================================================

correct_mask   = all_preds == all_labels
incorrect_mask = ~correct_mask

correct_indices   = np.where(correct_mask)[0]
incorrect_indices = np.where(incorrect_mask)[0]

def show_prediction_examples(indices, title, n=8):
    """Display sample predictions."""
    chosen = np.random.choice(indices, size=min(n, len(indices)), replace=False)
    fig, axes = plt.subplots(1, len(chosen), figsize=(3 * len(chosen), 4))
    if len(chosen) == 1:
        axes = [axes]
    for ax, idx in zip(axes, chosen):
        img = Image.open(all_paths[idx]).convert("RGB")
        ax.imshow(img)
        actual    = IDX_TO_EMOTION[all_labels[idx]]
        predicted = IDX_TO_EMOTION[all_preds[idx]]
        conf = all_probs[idx][all_preds[idx]] * 100
        color = "green" if all_preds[idx] == all_labels[idx] else "red"
        ax.set_title(f"Actual: {actual}\nPred: {predicted}\nConf: {conf:.1f}%",
                     fontsize=9, color=color)
        ax.axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

print(f"Correct predictions  : {len(correct_indices)} / {len(all_labels)}")
print(f"Incorrect predictions: {len(incorrect_indices)} / {len(all_labels)}")

np.random.seed(RANDOM_SEED)
show_prediction_examples(correct_indices,   "Correctly Classified Examples", n=8)
show_prediction_examples(incorrect_indices, "Incorrectly Classified Examples", n=8)
# ============================================================
# 11a. Check for demographic data availability
# ============================================================

print("=" * 60)
print("FAIRNESS ANALYSIS - Demographic Data Check")
print("=" * 60)

print(f"\nTrain CSV columns: {list(train_df.columns)}")
print(f"Test  CSV columns: {list(test_df.columns)}")

if HAS_DEMOGRAPHICS:
    print(f"\n[OK] Demographic columns found: {demographic_cols_test}")
    print("   Proceeding with group-wise fairness analysis.")
else:
    print("\n" + "-" * 60)
    print("LIMITATION: No Demographic Columns Found")
    print("-" * 60)
    print("The provided CSV files contain only 'image' and 'label' columns.")
    print("No gender, age, race, ethnicity, or other demographic attributes")
    print("are available for group-based fairness analysis.")
    print("\nConsequently:")
    print("  - Full demographic fairness analysis CANNOT be performed.")
    print("  - The framework below is structured and ready-to-use for when")
    print("    reliable demographic annotations are provided.")
    print("  - We provide per-CLASS fairness analysis as a proxy, examining")
    print("    whether certain emotion classes are systematically harder.")
    print("\n[NOTE] We do NOT generate demographic labels using AI models, as")
    print("predicted demographics would introduce additional noise and bias.")
# ============================================================
# 11b. Group-wise fairness analysis (when demographics are available)
# ============================================================

def compute_group_metrics(labels, preds, group_labels, group_name):
    """
    Compute per-group metrics for fairness analysis.
    
    Args:
        labels: Ground truth labels
        preds: Model predictions
        group_labels: Demographic group label for each sample
        group_name: Name of the demographic attribute
    
    Returns:
        DataFrame with group-wise metrics
    """
    unique_groups = sorted(set(group_labels))
    rows = []
    
    for group in unique_groups:
        mask = np.array(group_labels) == group
        g_labels = labels[mask]
        g_preds  = preds[mask]
        n = mask.sum()
        
        if n == 0:
            continue
        
        acc     = accuracy_score(g_labels, g_preds)
        prec    = precision_score(g_labels, g_preds, average="macro", zero_division=0)
        rec     = recall_score(g_labels, g_preds, average="macro", zero_division=0)
        f1_m    = f1_score(g_labels, g_preds, average="macro", zero_division=0)
        f1_w    = f1_score(g_labels, g_preds, average="weighted", zero_division=0)
        
        rows.append({
            "Attribute": group_name,
            "Group": group,
            "Samples": n,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Macro F1": round(f1_m, 4),
            "Weighted F1": round(f1_w, 4),
        })
    
    df = pd.DataFrame(rows)
    return df


if HAS_DEMOGRAPHICS:
    # Link test CSV to predictions
    # Build a mapping from image filename to demographic attributes
    for col in demographic_cols_test:
        # Map filename -> demographic value
        demo_map = dict(zip(test_df['image'], test_df[col]))
        
        # Get demographic labels for evaluated samples
        group_labels = []
        for path in all_paths:
            fname = os.path.basename(path)
            group_labels.append(demo_map.get(fname, "Unknown"))
        
        # Compute group metrics
        fairness_df = compute_group_metrics(all_labels, all_preds, group_labels, col)
        
        print(f"\n{'=' * 60}")
        print(f"FAIRNESS ANALYSIS - {col.upper()}")
        print(f"{'=' * 60}")
        display(fairness_df)
        
        # Performance gap
        best_f1  = fairness_df["Macro F1"].max()
        worst_f1 = fairness_df["Macro F1"].min()
        gap = best_f1 - worst_f1
        best_group  = fairness_df.loc[fairness_df["Macro F1"].idxmax(), "Group"]
        worst_group = fairness_df.loc[fairness_df["Macro F1"].idxmin(), "Group"]
        
        print(f"\n  Highest-performing group : {best_group} (Macro F1 = {best_f1:.4f})")
        print(f"  Lowest-performing group  : {worst_group} (Macro F1 = {worst_f1:.4f})")
        print(f"  Performance gap          : {gap:.4f}")
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(fairness_df))
        width = 0.3
        ax.bar(x - width/2, fairness_df["Accuracy"],  width, label="Accuracy",  color="#4C72B0")
        ax.bar(x + width/2, fairness_df["Macro F1"],  width, label="Macro F1",  color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(fairness_df["Group"], rotation=45)
        ax.set_ylabel("Score")
        ax.set_title(f"Fairness Analysis - {col.title()}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

else:
    print("\n[INFO] Skipping demographic group-wise fairness analysis (no demographics available).")
# ============================================================
# 11c. Per-CLASS fairness analysis (always available)
# ============================================================
# Even without demographic data, we can examine per-class performance
# disparities as a proxy for fairness analysis.

print("\n" + "=" * 60)
print("PER-CLASS FAIRNESS ANALYSIS (Proxy)")
print("=" * 60)
print("Since demographic annotations are not available, we analyse")
print("per-class performance disparities as a proxy.\n")

class_fairness_rows = []
for i, emo in enumerate(EMOTION_NAMES):
    mask = all_labels == i
    n_samples = mask.sum()
    if n_samples == 0:
        continue
    class_acc = (all_preds[mask] == all_labels[mask]).mean()
    # FNR = 1 - recall for this class
    class_recall = per_class_rec[i]
    fnr = 1.0 - class_recall
    
    class_fairness_rows.append({
        "Emotion": emo,
        "Samples": int(n_samples),
        "Accuracy": round(class_acc, 4),
        "Precision": round(per_class_prec[i], 4),
        "Recall": round(class_recall, 4),
        "F1-Score": round(per_class_f1[i], 4),
        "FNR": round(fnr, 4),
    })

class_fairness_df = pd.DataFrame(class_fairness_rows)
display(class_fairness_df)

best_class_f1  = class_fairness_df["F1-Score"].max()
worst_class_f1 = class_fairness_df["F1-Score"].min()
gap = best_class_f1 - worst_class_f1
best_emo  = class_fairness_df.loc[class_fairness_df["F1-Score"].idxmax(), "Emotion"]
worst_emo = class_fairness_df.loc[class_fairness_df["F1-Score"].idxmin(), "Emotion"]

print(f"\n  Best per-class F1  : {best_emo} ({best_class_f1:.4f})")
print(f"  Worst per-class F1 : {worst_emo} ({worst_class_f1:.4f})")
print(f"  Performance gap    : {gap:.4f}")
# ============================================================
# 12a. Mitigation via class-aware sample weighting
# ============================================================
# When demographic data IS available, replace class-based weights with
# group-aware or group x class-aware weights.

# NOTE: The baseline model above already uses class-weighted loss.
# Here we demonstrate an ADDITIONAL mitigation: oversampling minority classes
# using WeightedRandomSampler (complementary to weighted loss).

ENABLE_MITIGATION = True  # Set to False to skip this section

if ENABLE_MITIGATION:
    print("=" * 60)
    print("FAIRNESS MITIGATION EXPERIMENT")
    print("Method: Class-Balanced Oversampling (WeightedRandomSampler)")
    print("=" * 60)
    
    # Compute per-sample weights for oversampling
    train_targets = full_train_dataset.targets[train_indices]
    class_counts_train = np.bincount(train_targets, minlength=NUM_CLASSES)
    sample_weights = 1.0 / class_counts_train[train_targets]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    balanced_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    # New DataLoader with balanced sampling
    balanced_train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Build a fresh model
    mit_model, _ = build_resnet18_model()
    mit_model = mit_model.to(DEVICE)
    
    # Use standard (unweighted) CE since the sampler handles balance
    mit_criterion = nn.CrossEntropyLoss()
    mit_optimizer = optim.AdamW(mit_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        mit_optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    
    MIT_EPOCHS = min(15, NUM_EPOCHS)  # Shorter experiment
    MIT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "mitigated_resnet18.pth")
    
    best_mit_f1 = 0.0
    mit_patience_counter = 0
    
    print(f"\nTraining mitigated model for {MIT_EPOCHS} epochs...")
    print("-" * 40)
    
    for epoch in range(1, MIT_EPOCHS + 1):
        # Train
        t_loss, t_acc, t_f1 = train_one_epoch(
            mit_model, balanced_train_loader, mit_criterion, mit_optimizer, DEVICE
        )
        # Validate
        v_loss, v_acc, v_f1 = validate(
            mit_model, val_loader, mit_criterion, DEVICE
        )
        mit_scheduler.step(v_f1)
        
        print(f"  Epoch {epoch}/{MIT_EPOCHS} - "
              f"Train: loss={t_loss:.4f} acc={t_acc:.4f} F1={t_f1:.4f} | "
              f"Val: loss={v_loss:.4f} acc={v_acc:.4f} F1={v_f1:.4f}")
        
        if v_f1 > best_mit_f1:
            best_mit_f1 = v_f1
            torch.save({"model_state_dict": mit_model.state_dict()}, MIT_MODEL_PATH)
            mit_patience_counter = 0
        else:
            mit_patience_counter += 1
            if mit_patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break
    
    print(f"\nBest mitigated model Val Macro F1: {best_mit_f1:.4f}")
else:
    print("Mitigation experiment skipped (ENABLE_MITIGATION = False).")
# ============================================================
# 12b. Compare baseline vs mitigated model
# ============================================================
if ENABLE_MITIGATION:
    # Evaluate mitigated model on test set
    mit_ckpt = torch.load(MIT_MODEL_PATH, map_location=DEVICE, weights_only=False)
    mit_eval_model = models.resnet18(weights=None)
    mit_eval_model.fc = nn.Linear(mit_eval_model.fc.in_features, NUM_CLASSES)
    mit_eval_model.load_state_dict(mit_ckpt["model_state_dict"])
    mit_eval_model = mit_eval_model.to(DEVICE)
    mit_eval_model.eval()
    
    mit_preds = []
    mit_labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = mit_eval_model(images)
            mit_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            mit_labels_list.extend(labels.cpu().numpy())
    
    mit_preds = np.array(mit_preds)
    mit_labels_arr = np.array(mit_labels_list)
    
    # Compute mitigated model metrics
    mit_acc = accuracy_score(mit_labels_arr, mit_preds)
    mit_f1_macro = f1_score(mit_labels_arr, mit_preds, average="macro", zero_division=0)
    mit_per_class_f1 = f1_score(mit_labels_arr, mit_preds, average=None, zero_division=0)
    mit_worst_f1 = mit_per_class_f1.min()
    mit_gap = mit_per_class_f1.max() - mit_per_class_f1.min()
    
    # Baseline metrics (already computed)
    base_worst_f1 = per_class_f1.min()
    base_gap = per_class_f1.max() - per_class_f1.min()
    
    # Comparison table
    comparison = pd.DataFrame({
        "Metric": ["Overall Accuracy", "Macro F1", "Worst-Class F1", "Performance Gap (Best-Worst F1)"],
        "Baseline": [round(test_acc, 4), round(test_f1_macro, 4), round(base_worst_f1, 4), round(base_gap, 4)],
        "Mitigated (Balanced Sampling)": [round(mit_acc, 4), round(mit_f1_macro, 4), round(mit_worst_f1, 4), round(mit_gap, 4)],
    })
    
    print("\n" + "=" * 60)
    print("BASELINE vs MITIGATED MODEL COMPARISON")
    print("=" * 60)
    display(comparison)
    
    print("\nInterpretation:")
    if mit_worst_f1 > base_worst_f1:
        print("  [OK] Balanced sampling improved worst-class F1.")
    elif mit_worst_f1 < base_worst_f1:
        print("  [NOTE] Balanced sampling did not improve worst-class F1.")
    else:
        print("  Worst-class F1 unchanged.")
    
    if mit_gap < base_gap:
        print("  [OK] Performance gap reduced.")
    else:
        print("  [NOTE] Performance gap did not decrease.")
# ============================================================
# 13a. Grad-CAM implementation
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation for ResNet-18.
    
    Hooks into the specified target layer to capture activations
    and gradients, then computes the weighted activation map.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            heatmap: Grad-CAM heatmap (H, W) normalised to [0, 1]
            predicted_class: The predicted class index
            confidence: Confidence score for predicted class
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
        
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients[0]    # (C, h, w)
        activations = self.activations[0]  # (C, h, w)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=[1, 2])  # (C,)
        
        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (h, w)
        cam = F.relu(cam)  # ReLU to keep only positive influence
        
        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = cam.cpu().numpy()
        cam = np.uint8(255 * cam)
        cam = np.array(Image.fromarray(cam).resize(
            (input_tensor.shape[3], input_tensor.shape[2]),
            Image.BILINEAR
        ))
        cam = cam.astype(np.float32) / 255.0
        
        return cam, predicted_class, confidence


# Target layer: the last convolutional block of ResNet-18 (layer4)
grad_cam = GradCAM(eval_model, eval_model.layer4[-1])
print("[OK] Grad-CAM initialised (target layer: layer4[-1])")
# ============================================================
# 13b. Grad-CAM visualisation helper
# ============================================================

def visualize_gradcam(image_path, model, grad_cam_obj, transform, device,
                      actual_label=None, figsize=(14, 4)):
    """
    Generate and display Grad-CAM for a single image.
    
    Shows: Original image | Grad-CAM heatmap | Overlay
    """
    # Load and preprocess
    orig_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    heatmap, pred_class, confidence = grad_cam_obj.generate(input_tensor)
    
    # Create overlay
    orig_np = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * orig_np + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Labels
    pred_emotion = IDX_TO_EMOTION[pred_class]
    actual_str = IDX_TO_EMOTION[actual_label] if actual_label is not None else "N/A"
    is_correct = (actual_label == pred_class) if actual_label is not None else None
    border_color = "green" if is_correct else ("red" if is_correct is not None else "gray")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original\nActual: {actual_str}", fontsize=11)
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPred: {pred_emotion} ({confidence*100:.1f}%)",
                      fontsize=11, color=border_color)
    axes[2].axis("off")
    
    status = "CORRECT" if is_correct else ("INCORRECT" if is_correct is not None else "")
    status_color = "green" if is_correct else "red"
    plt.suptitle(status, fontsize=13, fontweight="bold", color=status_color)
    plt.tight_layout()
    plt.show()

print("[OK] Grad-CAM visualisation function defined.")
# ============================================================
# 13c. Grad-CAM for correctly classified samples
# ============================================================
print("=" * 60)
print("GRAD-CAM - Correctly Classified Samples")
print("=" * 60)

np.random.seed(RANDOM_SEED)
correct_samples = np.random.choice(correct_indices, size=min(5, len(correct_indices)), replace=False)

for idx in correct_samples:
    visualize_gradcam(
        all_paths[idx], eval_model, grad_cam,
        val_test_transform, DEVICE,
        actual_label=all_labels[idx]
    )
# ============================================================
# 13d. Grad-CAM for incorrectly classified samples
# ============================================================
print("\n" + "=" * 60)
print("GRAD-CAM - Incorrectly Classified Samples")
print("=" * 60)

np.random.seed(RANDOM_SEED + 1)
if len(incorrect_indices) > 0:
    incorrect_samples = np.random.choice(incorrect_indices, size=min(5, len(incorrect_indices)), replace=False)
    for idx in incorrect_samples:
        visualize_gradcam(
            all_paths[idx], eval_model, grad_cam,
            val_test_transform, DEVICE,
            actual_label=all_labels[idx]
        )
else:
    print("No incorrect predictions found.")
# ============================================================
# 13e. Grad-CAM for one sample from each emotion class
# ============================================================
print("\n" + "=" * 60)
print("GRAD-CAM - One Sample Per Emotion Class")
print("=" * 60)

np.random.seed(RANDOM_SEED + 2)
for class_idx, emotion in IDX_TO_EMOTION.items():
    class_mask = all_labels == class_idx
    class_indices_arr = np.where(class_mask)[0]
    if len(class_indices_arr) == 0:
        continue
    sample_idx = np.random.choice(class_indices_arr)
    print(f"\n--- {emotion} ---")
    visualize_gradcam(
        all_paths[sample_idx], eval_model, grad_cam,
        val_test_transform, DEVICE,
        actual_label=all_labels[sample_idx]
    )
# ============================================================
# 14. Save the final model with complete metadata
# ============================================================

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# The best model checkpoint was already saved during training (Section 8).
# Here we verify it and also save a JSON config for easy loading.

# Verify the checkpoint exists
assert os.path.isfile(BEST_MODEL_PATH), f"Best model not found at {BEST_MODEL_PATH}"

# Load and verify
saved_ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)

print("=" * 60)
print("FINAL MODEL SAVE VERIFICATION")
print("=" * 60)
print(f"Model path      : {os.path.abspath(BEST_MODEL_PATH)}")
print(f"File size        : {os.path.getsize(BEST_MODEL_PATH) / 1e6:.2f} MB")
print(f"Epoch            : {saved_ckpt['epoch']}")
print(f"Val Accuracy     : {saved_ckpt['val_acc']:.4f}")
print(f"Val Macro F1     : {saved_ckpt['val_macro_f1']:.4f}")
print(f"Weights source   : {'VGGFace2' if saved_ckpt['config']['using_vggface2'] else 'ImageNet (FALLBACK)'}")
print(f"Class mapping    : {saved_ckpt['class_mapping']}")

# Save a separate JSON config for easy reference
config_path = os.path.join(MODEL_SAVE_DIR, "model_config.json")
config_to_save = {
    "model_architecture": "ResNet-18",
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "normalize_mean": NORMALIZE_MEAN,
    "normalize_std": NORMALIZE_STD,
    "using_vggface2": USING_VGGFACE2,
    "class_to_idx": EMOTION_TO_IDX,
    "idx_to_class": {str(k): v for k, v in IDX_TO_EMOTION.items()},
    "best_epoch": int(saved_ckpt['epoch']),
    "val_accuracy": float(saved_ckpt['val_acc']),
    "val_macro_f1": float(saved_ckpt['val_macro_f1']),
    "checkpoint_path": os.path.abspath(BEST_MODEL_PATH),
}

with open(config_path, "w") as f:
    json.dump(config_to_save, f, indent=2)

print(f"\n[OK] Model config saved at: {os.path.abspath(config_path)}")
print(f"\n[OK] Best model successfully saved at: {os.path.abspath(BEST_MODEL_PATH)}")
# ============================================================
# 15a. Reusable inference function
# ============================================================

def load_model_for_inference(checkpoint_path, device=None):
    """
    Load the trained model and config from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load to (auto-detects if None)
    
    Returns:
        model: Loaded model in eval mode
        config: Configuration dictionary
        idx_to_class: Index-to-emotion mapping
        transform: Preprocessing transform for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    idx_to_class = {int(k): v for k, v in ckpt["class_mapping"].items()}
    
    # Build model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Build transform
    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["normalize_mean"], std=config["normalize_std"]),
    ])
    
    print(f"[OK] Model loaded from: {checkpoint_path}")
    print(f"   Architecture: ResNet-18, Classes: {config['num_classes']}")
    print(f"   Weights: {'VGGFace2' if config['using_vggface2'] else 'ImageNet (FALLBACK)'}")
    print(f"   Device: {device}")
    
    return model, config, idx_to_class, transform


def predict_emotion(image_path, model, transform, idx_to_class, device=None,
                    show_gradcam=True):
    """
    Predict emotion for a single image.
    
    Args:
        image_path: Path to the input image
        model: Loaded model
        transform: Preprocessing transform
        idx_to_class: Index-to-emotion mapping
        device: Device
        show_gradcam: Whether to generate and display Grad-CAM
    
    Returns:
        predicted_emotion (str), confidence (float), all_probs (dict)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load and preprocess
    orig_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
    
    pred_idx = probs.argmax().item()
    pred_emotion = idx_to_class[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs_dict = {idx_to_class[i]: round(probs[i].item(), 4) for i in range(len(probs))}
    
    # Display results
    print(f"\n{'=' * 40}")
    print(f"PREDICTION RESULT")
    print(f"{'=' * 40}")
    print(f"Image     : {image_path}")
    print(f"Predicted : {pred_emotion}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nAll probabilities:")
    for emo, prob in sorted(all_probs_dict.items(), key=lambda x: -x[1]):
        bar_len = int(prob * 30)
        bar = '#' * bar_len
        print(f"  {emo:>10s}: {prob*100:6.2f}% {bar}")
    
    # Show image
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(orig_img)
    ax.set_title(f"{pred_emotion} ({confidence*100:.1f}%)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Optional Grad-CAM
    if show_gradcam:
        gc = GradCAM(model, model.layer4[-1])
        input_tensor_gc = transform(orig_img).unsqueeze(0).to(device)
        heatmap, _, _ = gc.generate(input_tensor_gc)
        
        orig_resized = np.array(orig_img.resize((input_tensor_gc.shape[3], input_tensor_gc.shape[2]))) / 255.0
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        overlay = np.clip(0.5 * orig_resized + 0.5 * heatmap_colored, 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_resized)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM", fontsize=12)
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay - {pred_emotion}", fontsize=12)
        axes[2].axis("off")
        plt.suptitle("Grad-CAM Explanation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
    
    return pred_emotion, confidence, all_probs_dict

print("[OK] Inference functions defined.")
# ============================================================
# 15b. Demo: Load the saved model and predict on a test image
# ============================================================

# Load model
inf_model, inf_config, inf_idx_to_class, inf_transform = load_model_for_inference(
    BEST_MODEL_PATH, device=DEVICE
)

# Pick a random test image for demonstration
demo_class_folder = os.path.join(TEST_IMG_DIR, "4")  # Happiness folder
demo_images = [f for f in os.listdir(demo_class_folder)
               if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}]
demo_image_path = os.path.join(demo_class_folder, demo_images[0])

print(f"\nDemo prediction on: {demo_image_path}")

pred_emotion, conf, probs = predict_emotion(
    demo_image_path,
    inf_model,
    inf_transform,
    inf_idx_to_class,
    device=DEVICE,
    show_gradcam=True
)
# ============================================================
# 15c. Predict on your own image
# ============================================================
# To predict on a new image, update the path below and run this cell.

# NEW_IMAGE_PATH = r"path/to/your/image.jpg"  # <-- Set your image path here
#
# pred_emotion, conf, probs = predict_emotion(
#     NEW_IMAGE_PATH,
#     inf_model,
#     inf_transform,
#     inf_idx_to_class,
#     device=DEVICE,
#     show_gradcam=True
# )

print("[INFO] Uncomment the code above and set NEW_IMAGE_PATH to predict on your own image.")
# ============================================================
# 16. Programmatic summary
# ============================================================

print("=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\n1. MODEL & WEIGHTS")
print(f"   Architecture        : ResNet-18")
print(f"   Pretrained weights  : {'VGGFace2' if USING_VGGFACE2 else 'ImageNet (FALLBACK)'}")
if not USING_VGGFACE2:
    print(f"   [WARNING] ImageNet weights were used because VGGFace2 checkpoint was unavailable.")
    print(f"             Results may improve with VGGFace2-pretrained weights.")

print(f"\n2. TEST SET PERFORMANCE")
print(f"   Accuracy            : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"   Macro F1-Score      : {test_f1_macro:.4f}")
print(f"   Weighted F1-Score   : {test_f1_weighted:.4f}")
print(f"   Macro Precision     : {test_prec:.4f}")
print(f"   Macro Recall        : {test_rec:.4f}")

print(f"\n3. FAIRNESS ANALYSIS")
if HAS_DEMOGRAPHICS:
    print(f"   Demographic data available - group-wise analysis performed.")
else:
    print(f"   No demographic annotations available in the dataset CSVs.")
    print(f"   Per-class analysis used as proxy:")
    print(f"     Best per-class F1  : {best_emo} ({best_class_f1:.4f})")
    print(f"     Worst per-class F1 : {worst_emo} ({worst_class_f1:.4f})")
    print(f"     Performance gap    : {best_class_f1 - worst_class_f1:.4f}")

if ENABLE_MITIGATION:
    print(f"\n4. MITIGATION EXPERIMENT")
    print(f"   Method: Class-Balanced Oversampling")
    print(f"   Baseline Macro F1    : {test_f1_macro:.4f}")
    print(f"   Mitigated Macro F1   : {mit_f1_macro:.4f}")
    print(f"   Baseline Worst F1    : {base_worst_f1:.4f}")
    print(f"   Mitigated Worst F1   : {mit_worst_f1:.4f}")

print(f"\n5. GRAD-CAM")
print(f"   Grad-CAM heatmaps generated for correct, incorrect, and per-class samples.")
print(f"   Target layer: layer4[-1] (final convolutional block of ResNet-18).")

print(f"\n6. SAVED MODEL")
print(f"   Path: {os.path.abspath(BEST_MODEL_PATH)}")
print(f"   Config: {os.path.abspath(config_path)}")
