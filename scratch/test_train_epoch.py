import time, os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    train_dir = r'D:\Research4\TestDb\DATASET\train'
    test_dir = r'D:\Research4\TestDb\DATASET\test'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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

    # Use num_workers=0 to avoid any Windows spawn overhead
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}", flush=True)

    # Load VGGFace2 backbone - strictly face weights
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=7).to(device)

    # Unfreeze block8 and logits for fast convergence
    for p in model.parameters():
        p.requires_grad = False
    for p in model.block8.parameters():
        p.requires_grad = True
    for p in model.logits.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)

    t0 = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += len(y)
        if (i+1) % 50 == 0:
            print(f"Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Running Acc: {correct/total*100:.1f}%", flush=True)

    train_acc = correct / total
    train_loss = running_loss / total

    # Quick test eval
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_correct += (out.argmax(dim=1) == y).sum().item()
            test_total += len(y)

    test_acc = test_correct / test_total
    print(f"Epoch 1 Time: {time.time()-t0:.1f}s | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%", flush=True)

if __name__ == '__main__':
    main()
