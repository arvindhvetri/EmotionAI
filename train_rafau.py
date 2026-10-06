import os
import time
import argparse
from pathlib import Path
import warnings

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RAFAUDataset(Dataset):
    def __init__(self, data_dir, all_aus, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.all_aus = sorted(list(all_aus))
        self.au_to_idx = {au: i for i, au in enumerate(self.all_aus)}
        
        label_file = os.path.join(data_dir, "RAFAU_label.txt")
        img_dir = os.path.join(data_dir, "aligned")
        
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                img_name = parts[0]
                
                # Convert 0007.jpg to 0007_aligned.jpg
                base_name = os.path.splitext(img_name)[0]
                aligned_name = f"{base_name}_aligned.jpg"
                img_path = os.path.join(img_dir, aligned_name)
                
                if not os.path.exists(img_path):
                    continue
                
                # Multi-hot encoding
                label_tensor = torch.zeros(len(self.all_aus))
                if len(parts) > 1 and parts[1] != "null":
                    aus = parts[1].split('+')
                    for au in aus:
                        if au in self.au_to_idx:
                            label_tensor[self.au_to_idx[au]] = 1.0
                            
                self.samples.append((img_path, label_tensor))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, labels

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    IMG_SIZE = 224
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    
    # First pass to find all unique AUs
    label_file = os.path.join(args.data_dir, "RAFAU_label.txt")
    all_aus = set()
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1 and parts[1] != "null":
                all_aus.update(parts[1].split('+'))
                
    NUM_CLASSES = len(all_aus)
    print(f"Found {NUM_CLASSES} unique Action Units (AUs).")
    
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD  = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    print("Loading dataset...")
    full_dataset = RAFAUDataset(args.data_dir, all_aus, transform=None)
    
    # Calculate positive weights for highly imbalanced Action Units
    print("Calculating class weights for imbalanced AUs...")
    all_labels = torch.stack([labels for _, labels in full_dataset.samples])
    num_positives = all_labels.sum(dim=0)
    num_negatives = len(all_labels) - num_positives
    # Avoid division by zero for AUs that might have 0 positive samples
    num_positives[num_positives == 0] = 1 
    pos_weight = num_negatives / num_positives
    pos_weight = pos_weight.to(device)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])
    
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            img_path, labels = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, labels
        def __len__(self):
            return len(self.subset)
            
    train_dataset = TransformWrapper(train_subset, train_transform)
    test_dataset = TransformWrapper(test_subset, test_transform)
    
    print(f"Train images: {len(train_dataset)}, Validation images: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Initializing InceptionResnetV1 with VGGFace2 Weights...")
    from facenet_pytorch import InceptionResnetV1
    
    # Load VGGFace2 pretrained model
    model = InceptionResnetV1(pretrained='vggface2', classify=True)
    
    # Replace the final classification layer (logits) from 8631 classes to 76 AUs
    model.logits = nn.Linear(512, NUM_CLASSES)
    model = model.to(device)
    
    # Improved Loss with positive class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    
    # Learning rate scheduler based on F1 Score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0.0
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_vggface2_raf_au.pth")
    
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            if (i+1) % 20 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_dataset)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Step the learning rate scheduler
        scheduler.step(f1_macro)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (Macro): {f1_macro:.4f}")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model with F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to RAF-AU directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
