from matplotlib import pyplot as plt
import ctconfig as ctc
import torch
from torch import nn
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import time
import os

trial = 3
dest = f"modelstates/{trial}/"
if os.path.isdir(dest) is False:
    os.mkdir(dest)

ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/mat_norm32/Training/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/mat_norm32/Test/")
ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/mat_norm32/Validation/")
print("dataset size", len(ct_train_dataset))

ct_train_dataloader = DataLoader(ct_train_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)
ct_test_dataloader = DataLoader(ct_test_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)
ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)
print("dataloader size", len(ct_train_dataloader))

for image, label, file_name in ct_train_dataloader:
    print("Shape of image [N, C, H, W]: ", image.shape)
    print("Shape of label: ", label.shape, label.dtype)
    break

images, labels, file_names = next(iter(ct_train_dataloader))
for idx in range(0, 0):
    plt.subplot(1, 2, 1)
    plt.imshow(images[idx], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(labels[idx], cmap="gray")
    plt.title(file_names[idx])
    plt.show()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = ctu.NeuralNetwork().to(device)
print(model)
# model.load_state_dict(torch.load("modelstates9/model_1.pth"))
# model.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1} - {time.asctime(time.localtime(time.time()))}\n-------------------------------")
    # print(model.parameters())
    ctu.train(ct_train_dataloader, model, loss_fn, optimizer, device)
    # print(model.parameters())
    ctu.test(ct_test_dataloader, model, loss_fn, device)

    torch.save(model.state_dict(), f"modelstates/{trial}/model_{t}.pth")
    print("Saved PyTorch Model State to model.pth")
print("Done!")


