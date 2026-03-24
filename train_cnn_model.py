"""
============================================================
  SmartCrop — Disease Detection CNN Retraining Script
  Model: PyTorch CNN
  Dataset: PlantVillage (38 classes)
============================================================
HOW TO USE:
  1. Place your Train/ folder in C:\mini_project-main\
  2. Run: python train_cnn_model.py
  3. New model saved as: models/cnn_model.pth
============================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────
TRAIN_DIR    = "Train"
OUTPUT_DIR   = "models"
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 0.001
VAL_SPLIT    = 0.2
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────

# ─── CNN Model Definition ─────────────────────────────────
# Defined OUTSIDE __main__ so Windows multiprocessing can import it
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        flat_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── MAIN ─────────────────────────────────────────────────
# if __name__ == '__main__' is REQUIRED on Windows
# to prevent multiprocessing from spawning infinite processes
if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SmartCrop — CNN Disease Model Retraining")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ─── Step 1: Load Dataset ─────────────────────────────
    print("\n📂 Loading dataset...")

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"❌ Train folder not found at: {os.path.abspath(TRAIN_DIR)}\n"
            f"   Place your dataset folder here."
        )

    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    num_classes  = len(full_dataset.classes)
    print(f"   ✅ Found {len(full_dataset)} images across {num_classes} classes")
    print(f"   Classes: {full_dataset.classes}")

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    val_ds.dataset.transform = val_transforms

    # num_workers=0 is required on Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"   Train: {train_size} | Val: {val_size}")

    # ─── Step 2: Build Model ──────────────────────────────
    model = ImprovedCNN(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: ImprovedCNN | Parameters: {total_params:,}")

    # ─── Step 3: Loss, Optimizer, Scheduler ──────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # ─── Step 4: Training Loop ────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs on {DEVICE}...")
    print("-" * 60)

    best_val_acc     = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start = time.time()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - start

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            saved = "⭐ Best!"
        else:
            saved = ""

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
            f"{elapsed:.0f}s {saved}"
        )

    # ─── Step 5: Save Best Model ──────────────────────────
    print("\n💾 Saving best model...")
    model.load_state_dict(best_model_state)

    model_path       = os.path.join(OUTPUT_DIR, "cnn_model.pth")
    class_names_path = os.path.join(OUTPUT_DIR, "class_names.txt")

    torch.save(model.state_dict(), model_path)

    with open(class_names_path, "w") as f:
        for cls in full_dataset.classes:
            f.write(cls + "\n")

    print(f"   ✅ cnn_model.pth   → {os.path.abspath(model_path)}")
    print(f"   ✅ class_names.txt → {os.path.abspath(class_names_path)}")

    print("\n" + "=" * 60)
    print(f"  🎉 Training complete!")
    print(f"  Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Copy 'models/cnn_model.pth' to Frontend/models/")
    print("=" * 60)