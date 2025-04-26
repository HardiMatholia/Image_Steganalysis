import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

# ========== DEVICE ==========
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ========== SFTA FEATURE EXTRACTION ==========
def compute_fractal_dimension(image):
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    
    Z = image < 255
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def extract_sfta_features(gray_img, num_thresholds=3):
    thresholds = threshold_multiotsu(gray_img, classes=num_thresholds+1)
    binary_imgs = []

    for i in range(len(thresholds) + 1):
        if i == 0:
            binary = (gray_img < thresholds[0]).astype(np.uint8)
        elif i == len(thresholds):
            binary = (gray_img >= thresholds[-1]).astype(np.uint8)
        else:
            binary = ((gray_img >= thresholds[i-1]) & (gray_img < thresholds[i])).astype(np.uint8)
        binary_imgs.append(binary)

    features = []
    for binary in binary_imgs:
        fd = compute_fractal_dimension(binary)

        border = np.zeros_like(binary)
        for i in range(1, binary.shape[0]-1):
            for j in range(1, binary.shape[1]-1):
                if binary[i, j] == 1 and 0 in binary[i-1:i+2, j-1:j+2]:
                    border[i, j] = 1
        border_count = np.sum(border)

        labeled = label(binary)
        props = regionprops(labeled)
        mean_area = np.mean([p.area for p in props]) if props else 0

        features.extend([fd, border_count, mean_area])
    
    return features

# ========== DATASET CLASS ==========
class SFTADataset(Dataset):
    def __init__(self, stego_dir, cover_dir, max_samples=1000,
        feature_file='sfta_features.npy', label_file='sfta_labels.npy'):
        if os.path.exists(feature_file) and os.path.exists(label_file):
            self.features = np.load(feature_file)
            self.labels = np.load(label_file)
        else:
            self.features = []
            self.labels = []

            stego_files = sorted(os.listdir(stego_dir))[:max_samples]
            cover_files = sorted(os.listdir(cover_dir))[:max_samples]

            print("Extracting SFTA features...")
            for fname in tqdm(stego_files, desc="Stego"):
                img = cv2.imread(os.path.join(stego_dir, fname), cv2.IMREAD_GRAYSCALE)
                feat = extract_sfta_features(img)
                self.features.append(feat)
                self.labels.append(1)

            for fname in tqdm(cover_files, desc="Cover"):
                img = cv2.imread(os.path.join(cover_dir, fname), cv2.IMREAD_GRAYSCALE)
                feat = extract_sfta_features(img)
                self.features.append(feat)
                self.labels.append(0)

            self.features = np.array(self.features, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.float32)

            np.save(feature_file, self.features)
            np.save(label_file, self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# ========== CNN MODEL ==========
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        return self.net(x)

# ========== TRAINING ==========
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total

# ========== VALIDATION ==========
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total

# ========== MAIN ==========
def main():
    root = os.path.dirname(os.path.realpath(__file__))
    cover_dir = os.path.join(root, '..', 'DIP_Dataset', 'Cover10k')
    stego_dir = os.path.join(root, '..', 'DIP_Dataset', 'Stego10k')

    dataset = SFTADataset(stego_dir, cover_dir, max_samples=1000)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=16, shuffle=False)

    input_size = dataset[0][0].shape[0]
    model = CNNClassifier(input_size).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...\n")
    for epoch in trange(1, 11, desc="Epochs"):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main()
