import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from scipy.fft import dct
from sklearn.model_selection import train_test_split
import cv2

# ==============================
# GPU Gabor + Feature Extractor
# ==============================

def create_gabor_filters(device):
    print('in create filters..')
    filters = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        for sigma in [1, 2]:
            kernel = cv2.getGaborKernel((8, 8), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            filters.append(kernel_tensor)
    print('filters created..')
    return filters

# def apply_gabor_filters_gpu(img_tensor, filters):
#     responses = [torch.nn.functional.conv2d(img_tensor.unsqueeze(0).unsqueeze(0), f, padding=4).squeeze()
#                  for f in filters]
#     return responses

def apply_gabor_filters_gpu(img_tensor, filters):
    for i, f in enumerate(filters):
        assert f.device.type == 'cuda', f"Gabor filter {i} not on GPU!"
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    responses = [torch.nn.functional.conv2d(img_tensor, f, padding=4).squeeze() for f in filters]
    return responses

# def dct_block_subsample(img):
#     h, w = img.shape
#     dct_blocks = []
#     for i in range(0, h - 8 + 1, 8):
#         for j in range(0, w - 8 + 1, 8):
#             block = img[i:i+8, j:j+8].cpu().numpy()
#             dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
#             dct_blocks.append(dct_block)
#     return np.stack(dct_blocks)

# def dct_block_subsample(img_tensor):
#     h, w = img_tensor.shape
#     dct_blocks = []
#     for i in range(0, h - 8 + 1, 8):
#         for j in range(0, w - 8 + 1, 8):
#             block = img_tensor[i:i+8, j:j+8]
#             dct_block = dct(dct(block.cpu().numpy(), norm='ortho').T, norm='ortho')
#             dct_blocks.append(torch.tensor(dct_block, device=img_tensor.device))
#     return torch.stack(dct_blocks)

# def extract_histogram(dct_blocks, bins=16):
#     return np.histogram(dct_blocks.flatten(), bins=bins, range=(0, 50), density=True)[0]

# def extract_histogram(dct_blocks, bins=16):
#     assert dct_blocks.device.type == 'cuda', "DCT blocks not on GPU!"
#     flat_blocks = dct_blocks.flatten()
#     hist, _ = torch.histogram(flat_blocks, bins=bins, min=0, max=50)
#     return hist.float() / hist.sum()  # Normalize the histogram

# def extract_histogram(dct_blocks, bins=16):
#     # Flatten the dct_blocks and convert to tensor if necessary
#     flat_blocks = dct_blocks.flatten().clone().detach()
    
#     # Move to CPU for histogram calculation, as torch.histogram is not supported on CUDA
#     flat_blocks_cpu = flat_blocks.cpu()

#     # Use range to define the minimum and maximum
#     hist, _ = torch.histogram(flat_blocks_cpu, bins=bins, range=(0, 50), density=True)
#     return hist


# def extract_all_features(img_path, filters, device):
#     img = Image.open(img_path).convert("L")
#     img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=device) / 255.0
#     filtered_imgs = apply_gabor_filters_gpu(img_tensor, filters)

#     features = []
#     for f_img in filtered_imgs:
#         dct_blocks = dct_block_subsample(f_img)
#         hist = extract_histogram(dct_blocks)
#         features.append(hist)
#     return np.concatenate(features)

# def extract_all_features(img_path, filters, device):
#     img = Image.open(img_path).convert("L")
#     img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=device) / 255.0
#     filtered_imgs = apply_gabor_filters_gpu(img_tensor, filters)

#     features = []
#     for f_img in filtered_imgs:
#         dct_blocks = dct_block_subsample(f_img)
#         hist = extract_histogram(dct_blocks)
#         features.append(hist)
#     return np.concatenate(features)

def extract_all_features(img_path, filters, device):
    img = Image.open(img_path).convert("L")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=device) / 255.0
    filtered_imgs = apply_gabor_filters_gpu(img_tensor, filters)

    # Stack all the responses from Gabor filters
    features = [f_img.flatten().cpu().numpy() for f_img in filtered_imgs]
    return np.concatenate(features)

# ======================
# Preprocessing & Saving
# ======================

def preprocess_and_save_features(cover_folder, stego_folder, out_file, device):
    print('Preprocess start..')
    filters = create_gabor_filters(device)
    features, labels = [], []
    files = sorted(os.listdir(cover_folder))
    for f in files:
        cover_fp = os.path.join(cover_folder, f)
        stego_fp = os.path.join(stego_folder, f)

        cover_feat = extract_all_features(cover_fp, filters, device)
        stego_feat = extract_all_features(stego_fp, filters, device)

        features.extend([cover_feat, stego_feat])
        labels.extend([0, 1])
    print('preprocess end..')

    torch.save((torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long)), out_file)

# ==================
# Custom Dataset
# ==================

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================
# Model
# ==================

class FeatureClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._fc = nn.Sequential(
            nn.Linear(self.backbone._fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B = x.size(0)
        desired_features = 3 * 10 * 10
        if x.size(1) > desired_features:
            x = x[:, :desired_features]
        elif x.size(1) < desired_features:
            pad = torch.zeros((B, desired_features - x.size(1)), device=x.device)
            x = torch.cat([x, pad], dim=1)

        x = x.view(B, 3, 10, 10)
        x = torch.nn.functional.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
        return self.backbone(x)

# ==================
# Training Utilities
# ==================

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ====================
# Main Function
# ====================

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Change to your paths
    root = os.path.dirname(os.path.realpath(__file__))
    cover_folder = os.path.join(root, '..', 'DIP_Dataset', 'Cover10k')
    stego_folder = os.path.join(root, '..', 'DIP_Dataset', 'Stego10k')
    feature_file = 'features.pt'

    if not os.path.exists(feature_file):
        print("Extracting and saving features...")
        preprocess_and_save_features(cover_folder, stego_folder, feature_file, device)

    features, labels = torch.load(feature_file)
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
    train_data = FeatureDataset(features[train_idx], labels[train_idx])
    val_data = FeatureDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    print('data loaded classifier calling..')

    model = FeatureClassifier(input_dim=features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping()

    print('all set start training')

    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), "efficientnet_steg_model.pth")

if __name__ == "__main__":
    main()
