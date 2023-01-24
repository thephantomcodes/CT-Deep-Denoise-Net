import torch
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
        file_names = glob.glob(file_pattern)
        file_name = file_names[idx]
        mat = scio.loadmat(file_name)

        # Image generated from 1/4 projections
        image = np.array(mat['fbp_1'], dtype='float32')
        # Complete Image
        label = np.array(mat['fbp'], dtype='float32')

        # Image generated from 1/4 projections
        # image = np.array(mat['fbp_p'], dtype='float32')
        # Image generated from all available projections
        # label = np.array(mat['fbp'], dtype='float32')

        # # Image corrupted w/ Gaussian noise
        # image = np.array(mat['img_n'], dtype='float32')
        # Clean image
        # label = np.array(mat['img'], dtype='float32')
        # Residual image
        # label = image - np.array(mat['img'], dtype='float32')

        # image = image / np.max(image)
        # label = label / np.max(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        patch_shape = np.shape(image)
        input0 = torch.zeros(1, patch_shape[0], patch_shape[1])
        input0[0] = torch.from_numpy(image)
        output0 = torch.zeros(1, patch_shape[0], patch_shape[1])
        output0[0] = torch.from_numpy(label)
        return input0, output0, file_name
