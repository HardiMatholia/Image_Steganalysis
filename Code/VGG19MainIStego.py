import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import json

# Configuration
IMG_SIZE = 224  
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

# Dataset paths
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', 'DIP_Dataset') # Set the path to IStego100k Dataset directory

COVER_PATH = os.path.join(data_dir, 'Cover') 
STEGO_PATH = os.path.join(data_dir, 'Stego')
SS_TEST_PATH = os.path.join(data_dir, 'SS_test') # Set the path to test directory and json file for ground-truth labels
JSON_PATH = os.path.join(data_dir, 'SameSourceTestSetLabels.json')

# Get the labels from the json file
with open(JSON_PATH, 'r') as f:
    test_labels = json.load(f)

# Dataset class for train images
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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset and split
dataset = StegoDataset(COVER_PATH, STEGO_PATH, transform=transform)
indices = list(range(len(dataset)))
labels = np.array(dataset.labels)

train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

# Load Test images from a different test directory 
def load_test_images_dataset(test_folder, labels_dict, transform):
    images, labels = [], []
    for label, filenames in labels_dict.items():
        for filename in filenames:
            path = os.path.join(test_folder, filename)
            img = Image.open(path).convert('RGB')
            img = transform(img)
            images.append(img)
            labels.append(0 if label == 'cover' else 1)
    return torch.stack(images), torch.tensor(labels)

test_imgs, test_lbls = load_test_images_dataset(SS_TEST_PATH, test_labels, transform)
test_dataset = TensorDataset(test_imgs, test_lbls)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model: VGG19
model = models.vgg19(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 2)  
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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
        imgs = imgs.to(device)
        labels = labels.to(device)

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
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), correct / total

# Train loop
early_stopping = EarlyStopping(patience=PATIENCE, delta=0.001)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

# Final Evaluation
model.eval()
all_probs, all_preds, all_true = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # get prob for class '1' (stego)
        preds = torch.argmax(outputs, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_true.append(labels)

all_probs = torch.cat(all_probs)
all_preds = torch.cat(all_preds)
all_true = torch.cat(all_true)

test_acc = (all_preds == all_true).sum().item() / len(all_true)
test_auc = roc_auc_score(all_true.numpy(), all_probs.numpy())

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test AUC Score: {test_auc:.4f}")
