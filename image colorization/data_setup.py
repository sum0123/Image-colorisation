import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.color import rgb2lab
import os
from torchvision.io import read_image

import numpy as np


NUM_WORKERS = os.cpu_count()


def get_file_list(root):
    file_list = []
    for class_name in os.listdir(root):
        class_path = os.path.join(root, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                file_list.append((file_path))
    return file_list


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_path_list = get_file_list(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_path_list)

    # def load_image(self, index: int)
    #     "Opens an image via a path and returns it."
    #     image_path = self.img_path_list[index]
    #     return Image.open(image_path)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = read_image(img_path)

        image = image.permute(1, 2, 0)
        image = np.asarray(image)  # Convert the input data to a numpy array
        img_lab = rgb2lab(image)  # Convert the input image from RGB to LAB color space
        img_lab = (
            img_lab + 128
        ) / 255  # Normalize the LAB values so that they are in the range of 0 to 1
        img_ab = img_lab[:, :, 1:3]  # Get the "ab" channels from the LAB image
        img_ab = torch.from_numpy(
            img_ab.transpose((2, 0, 1))
        ).float()  # Convert the "ab" channels to a PyTorch tensor, bchw
        img_l = img_lab[:, :, 0]
        img_l = (
            torch.from_numpy(img_l).unsqueeze(0).float()
        )  # Convert the grayscale image to a PyTorch tensor and add a batch

        if self.transform:
            image = self.transform(img_l)
        if self.target_transform:
            label = self.target_transform(img_ab)
        return image, label


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int,
):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((192, 192)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((192, 192)),
        ]
    )
    # Use ImageFolder to create dataset(s)
    train_dataset = CustomImageDataset(
        img_dir=train_dir, transform=train_transforms, target_transform=train_transforms
    )
    test_dataset = CustomImageDataset(
        img_dir=test_dir, transform=test_transforms, target_transform=test_transforms
    )

    # Turn images into data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
