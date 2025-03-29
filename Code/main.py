import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define StegoCoverDataset class
class StegoCoverDataset(Dataset):
    def __init__(self, stego_dir=None, cover_dir=None, test_dir=None, transform=None):
        if test_dir:
            # Only for test data
            self.images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
            self.labels = [None] * len(self.images)  # No labels for test data
        else:
            # For training data
            self.stego_images = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir)][:100]
            self.cover_images = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir)][:100]
            self.images = self.stego_images + self.cover_images
            self.labels = [1] * len(self.stego_images) + [0] * len(self.cover_images)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx] if self.labels[idx] is not None else None
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Define the model using EfficientNetB4
class EfficientNetB4TransferModel(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB4TransferModel, self).__init__()
        # Load EfficientNetB4 pre-trained on ImageNet
        self.model = models.efficientnet_b4(pretrained=True)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the classifier layer for binary classification
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Setup transformations for data (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
])

# Setup datasets and dataloaders for both training and testing
current_dir = os.path.dirname(os.path.realpath(__file__))
stego_dir = os.path.join(current_dir, '..', 'DIP_Dataset', '100Stego')
cover_dir = os.path.join(current_dir, '..', 'DIP_Dataset', '100Cover')
test_dir = os.path.join(current_dir, '..', 'DIP_Dataset', 'test')

# Create the datasets for training and testing
train_dataset = StegoCoverDataset(stego_dir=stego_dir, cover_dir=cover_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = StegoCoverDataset(test_dir=test_dir, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def collate_fn(batch):
    # Custom collate function to handle test data where labels are None
    images = [item[0] for item in batch]
    return torch.stack(images, dim=0)

# Use the custom collate_fn in the test loader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
model = EfficientNetB4TransferModel(num_classes=1)  # Binary classification (0 or 1)
model = model.to(device)

# Use AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Loss function for binary classification
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification

# Train the model (same as before)
num_epochs = 10
model = model.to(device)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)  # Squeeze to remove extra dimension

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))  # Sigmoid for binary classification
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Test the model (skip loss calculation and handle None labels for test data)
model.eval()  # Set the model to evaluation mode
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Skip labels for test data and just predict
        predicted = torch.round(torch.sigmoid(outputs))  # Sigmoid for binary classification
        
        # Only count correct predictions during testing (no labels required)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

test_accuracy = correct_predictions / total_predictions * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
