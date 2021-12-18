from matplotlib import pyplot as plt
import ctconfig as ctc
import torch
from torch import nn
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu


ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Training/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Test/")
ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Validation/")

print(len(ct_train_dataset))

ct_train_dataloader = DataLoader(ct_train_dataset, batch_size=10, shuffle=True)
ct_test_dataloader = DataLoader(ct_test_dataset, batch_size=10, shuffle=True)
ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=10, shuffle=True)

print(len(ct_train_dataloader))

for X, y in ct_train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

images, labels = next(iter(ct_train_dataloader))
for idx in range(0, 0):
    plt.subplot(1, 2, 1)
    plt.imshow(images[idx], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(labels[idx], cmap="gray")
    plt.show()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = ctu.NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    ctu.train(ct_train_dataloader, model, loss_fn, optimizer, device)
    ctu.test(ct_test_dataloader, model, loss_fn, device)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
