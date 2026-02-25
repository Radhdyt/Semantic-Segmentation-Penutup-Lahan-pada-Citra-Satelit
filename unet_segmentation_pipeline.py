import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DATASET_FOLDER_NAME = 'dataset'

BASE_DIR = os.path.join(SCRIPT_DIR, DATASET_FOLDER_NAME)
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
MASK_DIR = os.path.join(BASE_DIR, 'masks')
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
IMAGE_SIZE = (256, 256)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLOR_MAP = {
    0: [255, 255, 0],
    1: [165, 42, 42],
    2: [0, 255, 0],
    3: [0, 0, 255]
}
NUM_CLASSES = len(COLOR_MAP)

class LandCoverDataset(Dataset):
    """
    Dataset loader kustom Pytorch untuk meload citra satelit dan mask grayscale.
    """
    def __init__(self, image_paths, mask_paths, image_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        
        self.img_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        image = self.img_transform(image)
        
        mask = mask.resize(self.image_size, resample=Image.NEAREST)
        
        mask_np = np.array(mask, dtype=np.int64) 
        
        mask_np[mask_np == 4] = 3 
        mask_tensor = torch.from_numpy(mask_np) 
        
        return image, mask_tensor

class DoubleConv(nn.Module):
    """Blok konvolusi x2 di sistem U-Net (Conv2d -> BatchNorm -> ReLU)"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Arsitektur U-Net original standar tanpa modifikasi.
    Encoder dan decoder tersambung dengan skip connections.
    """
    def __init__(self, in_channels=3, out_classes=4):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

def load_split_paths(split_file):
    """Membaca file txt split dan menghasilkan list absolute path untuk file hasil split.py di folder output"""
    with open(split_file, 'r') as f:
        basenames = [line.strip() for line in f.readlines() if line.strip()]
    
    SPLIT_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    images = [os.path.join(SPLIT_OUTPUT_DIR, f"{name}.jpg") for name in basenames]
    masks = [os.path.join(SPLIT_OUTPUT_DIR, f"{name}_m.png") for name in basenames]
    return images, masks

def train_model():
    train_file = os.path.join(BASE_DIR, 'train.txt')
    val_file = os.path.join(BASE_DIR, 'val.txt')
    test_file = os.path.join(BASE_DIR, 'test.txt')
    
    if not os.path.exists(train_file):
        print(f"File split tidak ditemukan di: {train_file}")
        return None

    X_train, y_train = load_split_paths(train_file)
    X_val, y_val = load_split_paths(val_file)
    X_test, y_test = load_split_paths(test_file)
    
    train_loader = DataLoader(LandCoverDataset(X_train, y_train, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LandCoverDataset(X_val, y_val, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_classes=NUM_CLASSES).to(DEVICE)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "unet_checkpoint.pth")
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Menemukan file checkpoint: {checkpoint_path}")
        print("Memuat status training terakhir untuk dilanjutkan...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"-> Bisa Dilanjutkan! Training akan di-resume mulai dari Epoch {start_epoch + 1}\n")

    print(f"Banyak Data => Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Mulai Training di Perangkat: {DEVICE} ...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"--- [Auto-Save] Model dan status belajar Epoch {epoch+1} berhasil diamankan! ---\n")
        
    print("Training Selesai!")
    return model

if __name__ == '__main__':
    trained_model = train_model()
