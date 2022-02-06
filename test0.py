import torch
from torch import nn
import ctconfig as ctc
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import numpy as np
import matplotlib.pyplot as plt


#ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Test/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/noisyPatches/Test/")
ct_test_dataloader = DataLoader(ct_test_dataset, batch_size=10, shuffle=True)
# print(ct_test_dataset[0][0])
device = "cpu"
input0 = torch.zeros(1, 10, 64, 64)
for batch, (image, label, file_name) in enumerate(ct_test_dataloader):
    image, label = image.to(device), label.to(device)
    input0[0] = image
    net = nn.Conv2d(10, 10, 5, stride=1, padding=2)
    output = net(input0)

    net = nn.BatchNorm2d(10)
    output = net(output)

    net = nn.ReLU()
    output = net(output)

    print(batch, np.shape(input0), np.shape(output))

    plt.subplot(1, 2, 1)
    plt.imshow(input0[0][0].detach(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(output[0][0].detach(), cmap="gray")
    plt.title(file_name)
    plt.show()

# model = ctu.NeuralNetwork().to(device)
#
# model.train()
# for batch, (image, label, file_name) in enumerate(ct_test_dataloader):
#     image, label = image.to(device), label.to(device)
#
#     # Compute prediction error
#     pred = model(image)
