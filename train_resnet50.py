import os
import time
import argparse
from pathlib import Path
from collections import Counter
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []
        
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            try:
                folder_label = int(folder_name)
                class_idx = folder_label - 1
            except ValueError:
                continue
            
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
        return image, label

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    IMG_SIZE = 224
    NUM_CLASSES = 7
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD  = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    print("Loading datasets...")
    train_dataset = RAFDBDataset(train_dir, transform=train_transform)
    test_dataset = RAFDBDataset(test_dir, transform=test_transform)
    print(f"Train images: {len(train_dataset)}, Test images: {len(test_dataset)}")
    
    # Class weights for imbalance
    class_counts = Counter(train_dataset.targets)
    total_samples = len(train_dataset)
    
    class_weights = [total_samples / class_counts[i] for i in range(NUM_CLASSES)]
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=total_samples, replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Initializing ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_f1 = 0.0
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_resnet50_rafdb.pth")
    
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
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        epoch_time = time.time() - start_time
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1_macro:.4f}")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model with F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"TestDb\DATASET")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
