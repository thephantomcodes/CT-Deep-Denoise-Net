from matplotlib import pyplot as plt
import ctconfig as ctc
import torch
from torch import nn
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import time
import os

trial = 2
dest = f"modelstates/fun_{trial}/"
if os.path.isdir(dest) is False:
    os.mkdir(dest)

data_loc = "mat_norm_fbp_quads"
ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + f"/data/{data_loc}/Training/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + f"/data/{data_loc}/Test/")
ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + f"/data/{data_loc}/Validation/")
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
#device = "cpu"
print(f"Using {device} device")

model = ctu.NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1e-4)

# adjust learning rate from 1e-3 to 1e-5 over 100 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.955, verbose=True)
scheduler.last_epoch = 100

start = 201
end = 401
if start > 1:
    print(f"loading model_{start-1}.pth")
    model.load_state_dict(torch.load(f"modelstates/fun_{trial}/model_{start-1}.pth"))
    model.eval()

epochs = range(start, end)
save_states = [50, 100, 150, 200]
print(f"last epoch {scheduler.last_epoch}")

for t in epochs:
    print(f"Epoch {t} - {time.asctime(time.localtime(time.time()))}\n-------------------------------")
    # print(model.parameters())
    ctu.train(ct_train_dataloader, model, loss_fn, optimizer, device)
    # print(model.parameters())
    ctu.test(ct_valid_dataloader, model, loss_fn, device)
    if t < scheduler.last_epoch:
        scheduler.step()
    print(f"Scheduler step {scheduler.get_last_lr()}, {scheduler.get_lr()}")

    if t in save_states:
        torch.save(model.state_dict(), f"modelstates/fun_{trial}/model_{t}.pth")
        print(f"Saved PyTorch Model State to model{t}.pth")
print("Done!")


