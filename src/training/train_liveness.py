import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_args():
    parser = argparse.ArgumentParser(description="Train Face Liveness (Anti-Spoofing) Model on ROCm")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset with 'real' and 'spoof' subfolders")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="liveness_model.pt", help="Path to save trained model weights")
    return parser.parse_args()

class LivenessModel(nn.Module):
    def __init__(self):
        super(LivenessModel, self).__init__()
        # Use a lightweight MobileNetV3 backbone for fast CPU/GPU inference
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # Modify the classifier head for binary classification (Real vs Spoof)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

def train(args):
    # Set device - ROCm maps natively to 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Define transforms for train and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Expecting dataset folder structure:
    # data_dir/
    #   train/
    #     real/
    #     spoof/
    #   val/
    #     real/
    #     spoof/
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Fallback: check if the direct directory contains real/spoof
        train_dir = args.data_dir
        val_dir = args.data_dir
        print("Warning: 'train' or 'val' subfolders not found. Loading entire directory as single split.")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Classes: {train_dataset.classes}")

    # Initialize model, loss, and optimizer
    model = LivenessModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1}/{args.epochs}] - Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

        # Save best weights
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model checkpoint to {args.save_path} with Val Acc: {best_acc:.2f}%")

    print("Training Complete!")

if __name__ == "__main__":
    args = get_args()
    train(args)
