import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1

def train_emotion_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START] Device: {device}", flush=True)

    train_dir = r'D:\Research4\TestDb\DATASET\train'
    test_dir = r'D:\Research4\TestDb\DATASET\test'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}", flush=True)

    # Strictly VGGFace2 face weights - NO ImageNet fallback
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=7).to(device)

    # Unfreeze block8, repeat_3 and logits for optimal adaptation to facial emotions
    for p in model.parameters():
        p.requires_grad = False
    for p in model.repeat_3.parameters():
        p.requires_grad = True
    for p in model.block8.parameters():
        p.requires_grad = True
    for p in model.logits.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    best_acc = 0.0
    save_path = os.path.join("models", "best_vggface2_emotion.pth")

    for epoch in range(1, 6):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

        train_acc = correct / total
        train_loss = running_loss / total
        scheduler.step()

        # Test evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_correct += (out.argmax(dim=1) == y).sum().item()
                test_total += len(y)

        test_acc = test_correct / test_total
        dt = time.time() - t0
        print(f"Epoch {epoch}/5 ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%", flush=True)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best emotion model to {save_path} (Acc: {best_acc*100:.2f}%)", flush=True)

    print(f"\n[DONE] Best Test Accuracy: {best_acc*100:.2f}%", flush=True)

if __name__ == '__main__':
    train_emotion_model()
