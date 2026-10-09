import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import HeartPartsDataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse


classes = ['LV', 'RV', 'LA', 'RA', 'Aorta']

# Paths to the dataset
image_dir = 'images'  # Directory containing images
mask_json_dir = 'masks_json_with_aorta'  # Path to the mask annotations

# Define transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
])

# Initialize the dataset and DataLoader
dataset = HeartPartsDataset(image_dir=image_dir, mask_json_dir=mask_json_dir, transform=transform, classes=classes, image_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def visualize_combined_and_individual(sample_idx=0):
    dataloader_iter = iter(dataloader)
    for _ in range(sample_idx + 1):  # skip to desired sample
        images, masks = next(dataloader_iter)

    image = images[0].permute(1, 2, 0).numpy()
    masks_np = masks[0].numpy()  # shape: [C, H, W]

    # Create combined mask: assign a unique color to each class
    # For visualization, assign a color per class
    colors = {
        'LV': [255, 0, 0],       # Red
        'RV': [0, 255, 0],       # Green
        'LA': [0, 0, 255],       # Blue
        'RA': [255, 255, 0],     # Yellow
        'Aorta': [255, 0, 255],  # Magenta
    }

    combined_mask = np.zeros((masks_np.shape[1], masks_np.shape[2], 3), dtype=np.uint8)
    for i, class_name in enumerate(classes):
        mask_i = masks_np[i]
        color = colors[class_name]
        for c in range(3):
            combined_mask[:, :, c] += (mask_i * color[c]).astype(np.uint8)
    combined_mask = np.clip(combined_mask, 0, 255)

    # Plot combined mask + original image side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(2, len(classes) + 1, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, len(classes) + 1, 2)
    plt.imshow(combined_mask)
    plt.title('Combined Mask')
    plt.axis('off')

    # Plot individual masks with colors
    for i, class_name in enumerate(classes):
        plt.subplot(2, len(classes) + 1, i + 3)
        single_mask_color = np.zeros_like(combined_mask)
        for c in range(3):
            single_mask_color[:, :, c] = (masks_np[i] * colors[class_name][c]).astype(np.uint8)
        plt.imshow(single_mask_color)
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'combined_and_individual_visualization_{timestamp}.png')
    plt.savefig(output_path)
    plt.show()
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_combined_and_individual(sample_idx=2)
    

