import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1

def train_balanced():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BALANCED TRAINING] Device: {device}", flush=True)

    train_dir = r'D:\Research4\TestDb\DATASET\train'
    test_dir = r'D:\Research4\TestDb\DATASET\test'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Calculate exact class weights for balanced per-class accuracy
    counts = [1290, 281, 717, 4772, 1982, 705, 2524]
    total = sum(counts)
    raw_weights = [total / (7.0 * c) for c in counts]
    # Moderate weights with power 0.6 to prevent over-amplifying noise while strongly boosting minority classes
    class_weights = torch.tensor([w ** 0.6 for w in raw_weights], dtype=torch.float32).to(device)
    print(f"Balanced Loss Weights: {[round(w.item(), 3) for w in class_weights]}", flush=True)

    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=7).to(device)

    # Unfreeze repeat_3, block8, logits
    for p in model.parameters():
        p.requires_grad = False
    for p in model.repeat_3.parameters():
        p.requires_grad = True
    for p in model.block8.parameters():
        p.requires_grad = True
    for p in model.logits.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3.5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-5)

    save_path = os.path.join("models", "best_vggface2_emotion.pth")
    best_balanced_acc = 0.0

    for epoch in range(1, 7):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        scheduler.step()

        # Evaluate balanced per-class accuracy on test set
        model.eval()
        class_correct = [0] * 7
        class_total = [0] * 7
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                for p, t in zip(preds, y):
                    class_total[t.item()] += 1
                    if p.item() == t.item():
                        class_correct[t.item()] += 1

        per_class_acc = [class_correct[i] / class_total[i] for i in range(7)]
        balanced_acc = sum(per_class_acc) / 7.0
        overall_acc = sum(class_correct) / sum(class_total)
        dt = time.time() - t0

        print(f"Epoch {epoch}/6 ({dt:.1f}s) | Balanced Acc: {balanced_acc*100:.2f}% | Overall Acc: {overall_acc*100:.2f}%", flush=True)
        print(f"  Per-class: {[f'{round(a*100, 1)}%' for a in per_class_acc]}", flush=True)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best balanced emotion model! (Balanced Acc: {best_balanced_acc*100:.2f}%)", flush=True)

if __name__ == '__main__':
    train_balanced()
