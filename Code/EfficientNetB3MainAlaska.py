import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import cv2

# Configuration
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Dataset paths
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', 'DIP_Dataset', 'Alaska_Dataset') # Set the path to Alaska Dataset directory
COVER_PATH = os.path.join(data_dir, 'Cover')
STEGO_PATH = os.path.join(data_dir, 'JUNIWARD') # Replace the Stego directory to UERD or JMiPOD for evaluating other Stego techniques

# move images for high-pass filter to GPU
def apply_module_on_cpu_tensor(module, x):
    device = next(module.parameters()).device
    return module(x.to(device)).cpu()

# High-pass filter
class HighPassFilter(nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        kernel = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.filter = nn.Conv2d(3, 3, 5, padding=2, bias=False, groups=3)
        self.filter.weight.data = kernel
        self.filter.weight.requires_grad = False

    def forward(self, x):
        return self.filter(x)

def gabor_filter(pil_img, ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5, psi=0):
    """
    Applies a Gabor filter to a grayscale version of a PIL image.

    Args:
        pil_img (PIL.Image): The input image.
        ksize: Size of the filter (odd integer).
        sigma: Standard deviation of the Gaussian envelope.
        theta: Orientation of the normal to the parallel stripes.
        lambd: Wavelength of the sinusoidal factor.
        gamma: Spatial aspect ratio.
        psi: Phase offset.

    Returns:
        PIL.Image: The Gabor-filtered image in RGB format.
    """
    img_gray = np.array(pil_img.convert("L"), dtype=np.float32)

    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)

    filtered_rgb = np.stack([filtered] * 3, axis=-1)  # Convert back to 3-channel RGB
    return Image.fromarray(filtered_rgb.astype(np.uint8))

# Dataset class
class StegoDataset(Dataset):
    def __init__(self, cover_folder, stego_folder, transform=None, highpass_filter=None, gabor_filter_fn=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.highpass_filter = highpass_filter
        self.gabor_filter_fn = gabor_filter_fn

        cover_images = os.listdir(cover_folder)
        stego_images = os.listdir(stego_folder)

        for img in cover_images:
            self.image_paths.append(os.path.join(cover_folder, img))
            self.labels.append(0)

        for img in stego_images:
            self.image_paths.append(os.path.join(stego_folder, img))
            self.labels.append(1)

        # Shuffle consistently
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Apply Gabor filter if provided before transform as it expects PIL image and not tensor
        if self.gabor_filter_fn:
            img = self.gabor_filter_fn(img)

        if self.transform:
            img = self.transform(img)

        # Apply highpass after if provided
        if self.highpass_filter:
            img = apply_module_on_cpu_tensor(self.highpass_filter, img.unsqueeze(0)).squeeze(0)
        
        return img, torch.tensor(label, dtype=torch.long)

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Optional preprocessing filters
highpass_filter = HighPassFilter().to(device)  # Set to None if not needed
# gabor_filter_fn = gabor_filter      # uncomment if want to apply gabor filter to images
gabor_filter_fn = None

# Dataset loading and splitting
dataset = StegoDataset(COVER_PATH, STEGO_PATH, transform=transform, highpass_filter=highpass_filter, gabor_filter_fn=gabor_filter_fn )
indices = list(range(len(dataset)))

train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=dataset.labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=np.array(dataset.labels)[temp_idx], random_state=42)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# Model: EfficientNetB3
model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = nn.Sequential(
    nn.Linear(model._fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2),  # Output logits for 2 classes
    nn.Softmax(dim=1)
)

# Move model to GPU
model = model.to(device)

# Loss and optimizer
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

# Training and Evaluation function
def train_epoch(model, dataloader, criterion, optimizer=None, train=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

# Train loop
early_stopper = EarlyStopping(patience=PATIENCE, delta=0.001)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, train=True)
    val_loss, val_acc = train_epoch(model, val_loader, criterion)
    scheduler.step(val_loss)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if early_stopper(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

# Final Evaluation
model.eval()
all_preds, all_probs, all_true = [], [], []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        outputs = model(imgs)
        probs = outputs[:, 1]
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