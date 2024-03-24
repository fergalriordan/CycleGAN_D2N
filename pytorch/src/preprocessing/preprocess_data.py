from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T 

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training transforms with data augmentations to artificially increase training set size
# Dimensions depend on whether a pre-trained encoder is used (224x224) or not (256x256) and therefore size is passed as a parameter to the function
def set_training_transforms(size, means, stds):
    transforms = A.Compose(
        [
            A.RandomResizedCrop(height=size, width=size, scale=(0.05, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),  
            A.OneOf([  # blurring
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.OneOf([  # distortion  
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.3),
            A.OneOf([  # colour augmentations/other
                A.CLAHE(clip_limit=2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3), 
                A.Sharpen(),
                A.RandomBrightnessContrast(),  
            ], p=0.1),
            A.Normalize(mean=means, std=stds, max_pixel_value=255),
            ToTensorV2(),
        ],
        #additional_targets={"image0": "image"}, 
    )
    return transforms

def set_val_transforms(means, stds):
    transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, max_pixel_value=255),
            ToTensorV2(),
        ],
    )
    return transforms

def load_image(image_path):
    with Image.open(image_path) as img:
        return np.array(img)

class DayNightDataset(Dataset):
    def __init__(self, root_night, root_day, size=None, day_transform=None, night_transform=None):
        self.root_night = root_night
        self.root_day = root_day
        #self.transform = transform
        self.day_transform = day_transform
        self.night_transform = night_transform
        self.size = size

        self.night_images = os.listdir(root_night)
        self.day_images = os.listdir(root_day)
        self.length_dataset = max(len(self.night_images), len(self.day_images))
        self.night_len = len(self.night_images)
        self.day_len = len(self.day_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        night_img = self.night_images[index % self.night_len]
        day_img = self.day_images[index % self.day_len]

        night_path = os.path.join(self.root_night, night_img)
        day_path = os.path.join(self.root_day, day_img)

        night_img = Image.open(night_path).convert("RGB")
        day_img = Image.open(day_path).convert("RGB")
        
        # iresize the images (either to training size or high-res test size)
        if self.size:
            night_img = night_img.resize((self.size, self.size), Image.ANTIALIAS) # Lanczos filter for resampling (my understanding is that this has been changed to 'LANCZOS' in newest PIL but Colab isn't updated yet)
            day_img = day_img.resize((self.size, self.size), Image.ANTIALIAS)
            
        night_img = np.array(night_img)
        day_img = np.array(day_img)

        day_img = self.day_transform(image=day_img)["image"]
        night_img = self.night_transform(image=night_img)["image"]

        #if self.transform:
        #    augmentations = self.transform(image=night_img, image0=day_img)
        #    night_img = augmentations["image"]
        #    day_img = augmentations["image0"]

        return night_img, day_img
        