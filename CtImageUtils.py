import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import ctconfig as ctc
import time


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        filt_sz = 3
        stride = 1
        padding = 1
        self.num_features = 32
        expansion = 10

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_features, kernel_size=filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
        )

        self.contracting_path_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(self.num_features, self.num_features*2, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features*2),
            nn.ReLU(),

            nn.Conv2d(self.num_features*2, self.num_features * 2, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(),

            nn.Conv2d(self.num_features*2, self.num_features * 2, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(),
        )

        self.contracting_path_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(self.num_features*2, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 4, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 4, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),
        )

        self.expanding_path_1 = nn.Sequential(
            nn.Conv2d(self.num_features * 4, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 4, self.num_features * 2, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 2, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
        )

        self.expanding_path_2 = nn.Sequential(
            nn.Conv2d(self.num_features * 4, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 4, self.num_features * 4, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(),

            nn.Conv2d(self.num_features * 4, self.num_features * 2, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.num_features*2, out_channels=self.num_features, kernel_size=filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.num_features, out_channels=1, kernel_size=filt_sz, stride=stride, padding=padding),
        )

    def forward(self, images):
        y0 = self.input_layer(images)
        c1 = self.contracting_path_1(y0)
        c2 = self.contracting_path_2(c1)
        e2 = self.expanding_path_2(c2)
        e3 = torch.cat([c1, e2], 1)
        e1 = self.expanding_path_1(e3)
        y2 = torch.cat([y0, e1], 1)
        return self.output_layer(y2)


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    input0 = torch.zeros(1, ctc.BATCH_SIZE, 32, 32)
    output0 = torch.zeros(1, ctc.BATCH_SIZE, 32, 32)
    for batch, (image, label, file_name) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        # input0 = []
        # input0[0] = image
        # output0[0] = label

        # Compute prediction error
        pred = model(image)

        # loss = loss_fn(pred, torch.reshape(label, (10, 4096)))
        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 160 == 0:
            # for x in range(0, 10):
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(image[x], cmap="gray")
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(label[x], cmap="gray")
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(pred[0][x].detach().numpy(), cmap="gray")
            #     plt.show()

            loss, current = loss.item(), batch * len(image)
            loss_sci = "{:e}".format(loss)
            print(f"loss: {loss_sci}  [{current:>5d}/{size:>5d}] - {time.asctime(time.localtime(time.time()))}")
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

        if loss == np.nan:
            exit()


def test(dataloader, model, loss_fn, device):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for label, label, file_name in dataloader:
            label, label = label.to(device), label.to(device)
            pred = model(label)
            test_loss += loss_fn(pred, label).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

