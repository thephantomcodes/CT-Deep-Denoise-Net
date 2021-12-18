from torch.utils.data import DataLoader, Dataset
from scipy import io as scio
import glob
import numpy as np


class CtImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        file_pattern = self.img_dir + "*.mat"
        files = glob.glob(file_pattern)
        return len(files)

    def __getitem__(self, idx):
        file_pattern = self.img_dir + "*.mat"
        files = glob.glob(file_pattern)
        file = files[idx]
        mat = scio.loadmat(file)
        # Image generated from 1/4 of available projections
        image = np.array(mat['fbp_0'], dtype='float32')
        # Image generated from all available projections
        label = np.array(mat['fbp_0'] + mat['fbp_1'] + mat['fbp_2'] + mat['fbp_3'], dtype='float32')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, file
