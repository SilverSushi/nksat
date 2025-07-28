import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, val=False, transform=None, target_transform=None):
        self.image_dir = os.path.join(dataset_path, 'images')
        self.mask_dir = os.path.join(dataset_path, 'masks')
        self.transform = transform
        self.target_transform = target_transform
        self.datapoints = []

        # Get class list and mapping
        self.classes = sorted(f for f in os.listdir(self.image_dir) if not f.startswith('.'))
        self.cls_to_int = {cls: i for i, cls in enumerate(self.classes)}
        self.int_to_cls = {i: cls for cls, i in self.cls_to_int.items()}

        # Count number of images per class
        self.class_image_counts = {}
        for cls in self.classes:
            img_folder = os.path.join(self.image_dir, cls)
            img_files = sorted(f for f in os.listdir(img_folder) if not f.startswith('.'))
            self.class_image_counts[cls] = len(img_files)

        # Compute average and find minority class indices
        counts = list(self.class_image_counts.values())
        avg_count = sum(counts) / len(counts)
        self.minority_class_indices = {
            self.cls_to_int[cls]
            for cls, count in self.class_image_counts.items()
            if count < avg_count
        }

        print(f"[INFO] Minority classes (by image count): {self.minority_class_indices}")

        # Gather all (image_path, mask_path, class_index) tuples
        for cls in self.classes:
            cls_int = self.cls_to_int[cls]
            img_folder = os.path.join(self.image_dir, cls)
            mask_folder = os.path.join(self.mask_dir, cls)
            img_files = sorted(f for f in os.listdir(img_folder) if not f.startswith('.'))

            # 80/20 split
            split_idx = int(0.8 * len(img_files))
            selected_files = img_files[split_idx:] if val else img_files[:split_idx]

            for fname in selected_files:
                img_path = os.path.join(img_folder, fname)
                mask_path = os.path.join(mask_folder, fname)
                self.datapoints.append((img_path, mask_path, cls_int))

        # Initialize two transform pipelines: light and strong
        self.light_aug = SegmentationJointTransform(strong=False)
        self.strong_aug = SegmentationJointTransform(strong=True)

    def __getitem__(self, idx):
        img_path, mask_path, cls_int = self.datapoints[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray(mask)

        #print('DEBUG: ', mask.shape, mask.dtype, torch.unique(mask))

        if self.transform:
          image = ToPILImage()(image.astype(np.uint8))
          image = self.transform(image)
          
        if self.target_transform:
          mask = self.target_transform(mask)

        if cls_int in self.minority_class_indices:
            image, mask = self.strong_aug(image, mask)
        else:
            image, mask = self.light_aug(image, mask)

        return image, mask, cls_int

    def __len__(self):
        return len(self.datapoints)
    
class SegmentationJointTransform:
    def __init__(self, image_size=(512, 512), strong=False):
        self.image_size = image_size
        self.strong = strong
        self.resize_img = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

    def __call__(self, img, mask):
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        if isinstance(mask, torch.Tensor):
            mask = T.ToPILImage()(mask.to(torch.uint8))

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # Random rotation
        angle = T.RandomRotation.get_params([-15, 15])
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)

        if self.strong:
            # Color jitter only applied to image
            img = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(img)

        img_tensor = T.ToTensor()(img)
        mask_tensor = T.PILToTensor()(mask).squeeze(0).long()

        return img_tensor, mask_tensor