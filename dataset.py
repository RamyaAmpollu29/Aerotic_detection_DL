import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon
from torchvision import transforms

class HeartPartsDataset(Dataset):
    def __init__(self, image_dir, mask_json_dir, transform=None, classes=None, image_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_json_dir = mask_json_dir
        self.transform = transform
        self.classes = classes if classes else []
        self.image_size = image_size

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_json_dir) if f.lower().endswith('.json')])

        assert len(self.image_files) == len(self.mask_files), "Number of images and mask files should match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size  # Store original size before resizing

        # Resize the image
        image = image.resize(self.image_size)

        mask_path = os.path.join(self.mask_json_dir, self.mask_files[idx])
        with open(mask_path, 'r') as f:
            mask_json = json.load(f)

        mask = np.zeros((len(self.classes), self.image_size[0], self.image_size[1]), dtype=np.uint8)

        for i, class_name in enumerate(self.classes):
            coords_list = mask_json.get(class_name, [])
            if not coords_list:
                continue

            xs, ys = zip(*coords_list)
            scale_x = self.image_size[1] / orig_width
            scale_y = self.image_size[0] / orig_height

            xs_scaled = np.array(xs) * scale_x
            ys_scaled = np.array(ys) * scale_y

            rr, cc = polygon(ys_scaled, xs_scaled, shape=mask.shape[1:])
            mask[i, rr, cc] = 1

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(mask).float()

        return image, mask_tensor
