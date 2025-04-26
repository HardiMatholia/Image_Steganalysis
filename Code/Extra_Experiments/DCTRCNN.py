import os
import random
import numpy as np
from PIL import Image
from scipy.fftpack import dct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ========== Helper Functions ==========

def load_grayscale(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32)

def blockwise_dct(img, block_size=8):
    h, w = img.shape
    h_crop, w_crop = h - h % block_size, w - w % block_size
    img = img[:h_crop, :w_crop]
    blocks = img.reshape(h_crop // block_size, block_size, -1, block_size).transpose(0, 2, 1, 3)
    dct_blocks = dct(dct(blocks, axis=2, norm='ortho'), axis=3, norm='ortho')
    return dct_blocks

def dct_blocks_to_tensor(dct_blocks):
    h_blocks, w_blocks, bh, bw = dct_blocks.shape
    dct_tensor = np.zeros((bh * bw, h_blocks, w_blocks), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            idx = i * bw + j
            dct_tensor[idx] = dct_blocks[:, :, i, j]
    return dct_tensor

def normalize_dct(dct_tensor):
    # Normalize DCT coefficients to range [-1, 1]
    return np.clip(dct_tensor / np.max(np.abs(dct_tensor)), -1, 1)

def extract_low_frequency_dct(img, block_size=8):
    # Extract low-frequency components of the DCT for feature extraction
    dct_blocks = blockwise_dct(img, block_size)
    low_freq_dct = dct_blocks[:, :, :4, :4]  # Keep only low-frequency components (top left 4x4 DCT block)
    return dct_blocks_to_tensor(low_freq_dct)

# ========== Dataset Class ==========

class DCTDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        gray = load_grayscale(path)
        dct_tensor = extract_low_frequency_dct(gray)  # Use low-frequency DCT coefficients
        dct_tensor = normalize_dct(dct_tensor)
        return torch.tensor(dct_tensor), torch.tensor(label, dtype=torch.long)

# ========== Custom CNN Model ==========

class CustomCNN(nn.Module):
    def __init__(self, in_channels=16):  # default set to 16
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ========== Main Script ==========

if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, '..', 'DIP_Dataset','Alaska_Dataset')
    cover_dir = os.path.join(data_dir, 'Cover')
    stego_dir = os.path.join(data_dir, 'UERD')

    # cover_dir = "./cover"
    # stego_dir = "./stego"

    # Load paths
    cover_paths = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith('.jpg')]
    stego_paths = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith('.jpg')]

    # Limit to 500 each
    cover_paths = sorted(cover_paths)[:500]
    stego_paths = sorted(stego_paths)[:500]

    image_paths = cover_paths + stego_paths
    labels = [0] * len(cover_paths) + [1] * len(stego_paths)

    # Shuffle
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)

    # Split into train/val/test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Datasets and Dataloaders
    train_dataset = DCTDataset(train_paths, train_labels)
    val_dataset = DCTDataset(val_paths, val_labels)
    test_dataset = DCTDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model, Optimizer, Loss
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(in_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

  # Train for More Epochs
    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            correct += (preds == batch_y).sum().item()  # Count correct predictions
            total += batch_y.size(0)  # Total number of samples
        val_accuracy = correct / total * 100
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Learning Rate Scheduling
    scheduler.step(avg_train_loss)
