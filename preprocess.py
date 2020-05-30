import os
import cv2
import torch
from torch.utils.data import Dataset

class LoadCelebFaces(Dataset):
    def __init__(self, transform=None):
        self.PATH = "../../Computer Vision/celeb_faces/data/img_align_celeba/img_align_celeba/"
        self.names = os.listdir(self.PATH)
        self.samples = [()] * len(self.names)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(self.PATH+name, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        img = cv2.resize(img, (50, 50))
        if self.transform:
            img = self.transform(img)
        self.samples[idx] = img

        return self.samples[idx]
