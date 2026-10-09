import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HeartPartsDataset
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

# Modified UNet to output 5 classes
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5):
        super(UNet, self).__init__()
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv1 = self.double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = self.double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = self.double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = self.double_conv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

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

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def down(self, in_c, out_c):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_c, out_c)
        )


# Paths
image_dir = 'images'
mask_json_dir = 'masks_json_with_aorta'

# Normalize with mean/std 0.5 (adjust if you have better estimates)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

classes = ['LV', 'RV', 'LA', 'RA', 'Aorta']

dataset = HeartPartsDataset(image_dir=image_dir, mask_json_dir=mask_json_dir, transform=transform, classes=classes)

# Reproducible split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(n_channels=3, n_classes=len(classes)).to(device)

# Optionally weight Aorta class more due to class imbalance
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def dice_loss_multiclass(pred, target, smooth=1e-6):
    pred_probs = torch.softmax(pred, dim=1)
    total_loss = 0.0
    num_classes = pred.shape[1]

    for c in range(num_classes):
        pred_c = pred_probs[:, c, :, :]
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2 * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice).mean()

    return total_loss / num_classes

def iou_score_multiclass(pred, target, smooth=1e-6):
    pred_class = torch.argmax(pred, dim=1)
    num_classes = pred.shape[1]
    iou = 0.0
    for c in range(num_classes):
        pred_c = (pred_class == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou += (intersection + smooth) / (union + smooth)
    return iou / num_classes

def visualize_multiclass_prediction(epoch, dataloader, classes):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        image = images[0:1].to(device)
        mask = masks[0:1]  # shape [1, C, H, W]
        mask_class = torch.argmax(mask, dim=1)[0].cpu().numpy()

        output = model(image)
        pred_class = torch.argmax(output, dim=1)[0].cpu().numpy()

        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5) + 0.5  # De-normalize for visualization

        colors = {
            0: [255, 0, 0],     # LV - red
            1: [0, 255, 0],     # RV - green
            2: [0, 0, 255],     # LA - blue
            3: [255, 255, 0],   # RA - yellow
            4: [255, 0, 255],   # Aorta - magenta
        }

        def color_mask(mask_class):
            h, w = mask_class.shape
            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            for c, color in colors.items():
                color_img[mask_class == c] = color
            return color_img

        mask_color = color_mask(mask_class)
        pred_color = color_mask(pred_class)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_color)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_color)
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'results/multiclass_prediction_epoch_{epoch+1}.png')
        plt.close()

    model.train()

def evaluate_model(dataloader):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            target = torch.argmax(masks, dim=1).long()

            outputs = model(images)
            loss_ce = criterion(outputs, target)
            loss_dice = dice_loss_multiclass(outputs, target)
            loss = loss_ce + loss_dice

            val_loss += loss.item()
            val_dice += (1 - loss_dice).item()
            val_iou += iou_score_multiclass(outputs, target).item()

    val_loss /= len(dataloader)
    val_dice /= len(dataloader)
    val_iou /= len(dataloader)

    model.train()
    return val_loss, val_dice, val_iou

def train_model(num_epochs=80):
    best_val_dice = 0.0
    train_losses, val_losses, val_dices, val_ious = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            target = torch.argmax(masks, dim=1).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion(outputs, target)
            loss_dice = dice_loss_multiclass(outputs, target)
            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_dice, val_iou = evaluate_model(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"Saved Best Model at Epoch {epoch+1}")

        if (epoch + 1) % 5 == 0:
            visualize_multiclass_prediction(epoch, val_loader, classes)

    # Plot losses and metrics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_dices, label='Val Dice')
    plt.plot(val_ious, label='Val IoU')
    plt.legend()
    plt.title('Validation Metrics')

    plt.savefig('results/training_metrics.png')
    plt.show()

if __name__ == "__main__":
    train_model()
