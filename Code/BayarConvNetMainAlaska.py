import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random

# Configuration
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

# Dataset paths
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', 'DIP_Dataset','Alaska_Dataset') # Set the path to Alaska Dataset directory
COVER_PATH = os.path.join(data_dir, 'Cover')
STEGO_PATH = os.path.join(data_dir, 'JUNIWARD')  # Replace the Stego directory to UERD or JMiPOD for evaluating other Stego techniques

# Dataset class
class StegoDataset(Dataset):
    def __init__(self, cover_folder, stego_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        cover_images = os.listdir(cover_folder)
        stego_images = os.listdir(stego_folder)

        for img in cover_images:
            self.image_paths.append(os.path.join(cover_folder, img))
            self.labels.append(0)

        for img in stego_images:
            self.image_paths.append(os.path.join(stego_folder, img))
            self.labels.append(1)

        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Data loading and splitting
dataset = StegoDataset(COVER_PATH, STEGO_PATH, transform=transform)
indices = list(range(len(dataset)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=dataset.labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=np.array(dataset.labels)[temp_idx], random_state=42)

train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# BayarConv2d Layer
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        with torch.no_grad():
            center = kernel_size // 2
            self.weights[:, :, center, center] = -1.0

    def forward(self, x):
        w = self.weights.clone()
        center = self.kernel_size // 2
        w[:, :, center, center] = 0
        w = w / w.sum(dim=[2, 3], keepdim=True)
        w[:, :, center, center] = -1
        return torch.nn.functional.conv2d(x, w, stride=1, padding=self.kernel_size // 2)

# BayarConvNet Architecture
class BayarConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bayar = BayarConv2d(1, 8)
        self.conv_block = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.bayar(x)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

# GPU setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = BayarConvNet().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# Training functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), correct / total

# Train loop
early_stopping = EarlyStopping(patience=PATIENCE, delta=0.0005)
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
    if early_stopping(val_loss):
        print("Early stopping triggered.")
        break

# Final Evaluation
model.eval()
all_preds, all_probs, all_true = [], [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_true.append(lbls.cpu())

all_preds = torch.cat(all_preds)
all_probs = torch.cat(all_probs)
all_true = torch.cat(all_true)

test_accuracy = (all_preds == all_true).sum().item() / len(all_true)
test_auc = roc_auc_score(all_true.numpy(), all_probs.numpy())

print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test AUC Score: {test_auc:.4f}")
