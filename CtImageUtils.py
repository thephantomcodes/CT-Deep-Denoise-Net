import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import ctconfig as ctc
import time
from pytorch_wavelets import DWTForward, DWTInverse


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, num_features=64, wave="haar"):
        super(NeuralNetwork, self).__init__()
        self.wave = wave
        self.fwd = DWTForward(wave=self.wave).to("cuda" if torch.cuda.is_available() else "cpu")
        self.inv = DWTInverse(wave=self.wave).to("cuda" if torch.cuda.is_available() else "cpu")
        # stage 0
        self.stage_0_0 = self.conv_bn_relu(num_features_in=in_channels, num_features_out=num_features)
        self.stage_0_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        self.stage_0_wt = DWTForward(wave=self.wave)
        self.stage_0_HH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_HH_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)
        self.stage_0_HL_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_HL_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)
        self.stage_0_LH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_LH_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)

        # stage 1
        self.stage_1_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_1_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 2)
        self.stage_1_wt = DWTForward(wave=self.wave)
        self.stage_1_HH_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_HH_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)
        self.stage_1_HL_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_HL_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)
        self.stage_1_LH_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_LH_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)

        # stage 2
        self.stage_2_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_2_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 4)
        self.stage_2_wt = DWTForward(wave=self.wave)
        self.stage_2_HH_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_HH_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)
        self.stage_2_HL_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_HL_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)
        self.stage_2_LH_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_LH_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)

        # stage 3
        self.stage_3_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_3_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 8)
        self.stage_3_wt = DWTForward(wave=self.wave)
        self.stage_3_HH_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_HH_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_HL_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_HL_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_LH_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_LH_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_LL_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_LL_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)

        # reconstruction
        self.stage_3_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 8,
                                                          num_features_out=num_features * 8)
        self.stage_3_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 8,
                                                          num_features_out=num_features * 4)

        self.stage_2_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 4,
                                                          num_features_out=num_features * 4)
        self.stage_2_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 4,
                                                          num_features_out=num_features * 2)

        self.stage_1_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 2,
                                                          num_features_out=num_features * 2)
        self.stage_1_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 2,
                                                          num_features_out=num_features)

        self.stage_0_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        self.stage_0_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)

        self.reconstruction_output = nn.Conv2d(num_features, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def conv_bn_relu(self, num_features_in, num_features_out):
        layers = []
        layers.append(nn.Conv2d(num_features_in, num_features_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(num_features_out))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def fwd_wavelet(self, x):
        Yh, Yl = self.fwd(x)
        return Yh, (Yl[0][:, :, 0], Yl[0][:, :, 1], Yl[0][:, :, 2])

    def inv_wavelet(self, ll, lh, hl, hh):
        h = [torch.stack((lh, hl, hh), dim=2)]
        return self.inv((ll, h))

    def forward(self, x, target=None):
        # stage 0
        stage_0_0 = self.stage_0_0(x)
        stage_0_1 = self.stage_0_1(stage_0_0)
        stage_0_LL, (stage_0_LH, stage_0_HL, stage_0_HH) = self.fwd_wavelet(stage_0_1)
        stage_0_LH_0 = self.stage_0_LH_0(stage_0_LH)
        stage_0_LH_1 = self.stage_0_LH_1(stage_0_LH_0)
        stage_0_HL_0 = self.stage_0_HL_0(stage_0_HL)
        stage_0_HL_1 = self.stage_0_HL_1(stage_0_HL_0)
        stage_0_HH_0 = self.stage_0_HH_0(stage_0_HH)
        stage_0_HH_1 = self.stage_0_HH_1(stage_0_HH_0)

        # stage 1
        stage_1_0 = self.stage_1_0(stage_0_LL)
        stage_1_1 = self.stage_1_1(stage_1_0)
        stage_1_LL, (stage_1_LH, stage_1_HL, stage_1_HH) = self.fwd_wavelet(stage_1_1)
        stage_1_LH_0 = self.stage_1_LH_0(stage_1_LH)
        stage_1_LH_1 = self.stage_1_LH_1(stage_1_LH_0)
        stage_1_HL_0 = self.stage_1_HL_0(stage_1_HL)
        stage_1_HL_1 = self.stage_1_HL_1(stage_1_HL_0)
        stage_1_HH_0 = self.stage_1_HH_0(stage_1_HH)
        stage_1_HH_1 = self.stage_1_HH_1(stage_1_HH_0)

        # stage 2
        stage_2_0 = self.stage_2_0(stage_1_LL)
        stage_2_1 = self.stage_2_1(stage_2_0)
        stage_2_LL, (stage_2_LH, stage_2_HL, stage_2_HH) = self.fwd_wavelet(stage_2_1)
        stage_2_LH_0 = self.stage_2_LH_0(stage_2_LH)
        stage_2_LH_1 = self.stage_2_LH_1(stage_2_LH_0)
        stage_2_HL_0 = self.stage_2_HL_0(stage_2_HL)
        stage_2_HL_1 = self.stage_2_HL_1(stage_2_HL_0)
        stage_2_HH_0 = self.stage_2_HH_0(stage_2_HH)
        stage_2_HH_1 = self.stage_2_HH_1(stage_2_HH_0)

        # stage 3
        stage_3_0 = self.stage_3_0(stage_2_LL)
        stage_3_1 = self.stage_3_1(stage_3_0)
        stage_3_LL, (stage_3_LH, stage_3_HL, stage_3_HH) = self.fwd_wavelet(stage_3_1)
        stage_3_LH_0 = self.stage_3_LH_0(stage_3_LH)
        stage_3_LH_1 = self.stage_3_LH_1(stage_3_LH_0)
        stage_3_HL_0 = self.stage_3_HL_0(stage_3_HL)
        stage_3_HL_1 = self.stage_3_HL_1(stage_3_HL_0)
        stage_3_HH_0 = self.stage_3_HH_0(stage_3_HH)
        stage_3_HH_1 = self.stage_3_HH_1(stage_3_HH_0)
        stage_3_LL_0 = self.stage_3_LL_0(stage_3_LL)
        stage_3_LL_1 = self.stage_3_LL_1(stage_3_LL_0)

        # reconstruction
        stage_3_reconstruction = self.inv_wavelet(stage_3_LL_1, stage_3_LH_1, stage_3_HL_1, stage_3_HH_1)
        # stage_3_transform[0] = stage_3_LL_1
        # stage_3_reconstruction = self.stage_3_wt.wavelet_n_rec(stage_3_transform)
        stage_3_LL_reconstruction_0 = self.stage_3_reconstruction_0(stage_3_reconstruction + stage_3_1)
        stage_3_LL_reconstruction_1 = self.stage_3_reconstruction_1(stage_3_LL_reconstruction_0)

        stage_2_reconstruction = self.inv_wavelet(stage_3_LL_reconstruction_1, stage_2_LH_1, stage_2_HL_1, stage_2_HH_1)
        # stage_2_transform[0] = stage_2_LL_1
        # stage_2_reconstruction = self.stage_2_wt.wavelet_n_rec(stage_2_transform)
        stage_2_LL_reconstruction_0 = self.stage_2_reconstruction_0(stage_2_reconstruction + stage_2_1)
        stage_2_LL_reconstruction_1 = self.stage_2_reconstruction_1(stage_2_LL_reconstruction_0)

        stage_1_reconstruction = self.inv_wavelet(stage_2_LL_reconstruction_1, stage_1_LH_1, stage_1_HL_1, stage_1_HH_1)
        # stage_1_transform[0] = stage_1_LL_1
        # stage_1_reconstruction = self.stage_1_wt.wavelet_n_rec(stage_1_transform)
        stage_1_LL_reconstruction_0 = self.stage_1_reconstruction_0(stage_1_reconstruction + stage_1_1)
        stage_1_LL_reconstruction_1 = self.stage_1_reconstruction_1(stage_1_LL_reconstruction_0)

        stage_0_reconstruction = self.inv_wavelet(stage_1_LL_reconstruction_1, stage_0_LH_1, stage_0_HL_1, stage_0_HH_1)
        # stage_0_transform[0] = stage_0_LL_1
        # stage_0_reconstruction = self.stage_0_wt.wavelet_n_rec(stage_0_transform)
        stage_0_LL_reconstruction_0 = self.stage_0_reconstruction_0(stage_0_reconstruction + stage_0_1)
        stage_0_LL_reconstruction_1 = self.stage_0_reconstruction_1(stage_0_LL_reconstruction_0)

        out = self.reconstruction_output(stage_0_LL_reconstruction_1) + x
        return out


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

        if batch % 10 == 0:
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

