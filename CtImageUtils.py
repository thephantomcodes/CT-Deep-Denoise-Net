import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import ctconfig as ctc


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        filt_sz = 3
        stride = 1
        padding = 1
        self.num_features = 50
        
        self.conv_batch_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_features, kernel_size=filt_sz, stride=stride, padding=padding+10),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(self.num_features, self.num_features, filt_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.num_features, out_channels=1, kernel_size=filt_sz, stride=stride, padding=padding),
            nn.ZeroPad2d(-10)
        )

    def forward(self, images):
        return self.conv_batch_relu_stack(images)


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

        if batch % 100 == 0:
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
            print(f"loss: {loss_sci}  [{current:>5d}/{size:>5d}]")
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

