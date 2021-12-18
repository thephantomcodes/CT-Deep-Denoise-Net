import numpy as np
from scipy import io as scio
import ctconfig as ctc
import torch
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu

model = ctu.NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Training/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Test/")
ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Validation/")

ct_train_dataloader = DataLoader(ct_train_dataset, batch_size=10, shuffle=True)
ct_test_dataloader = DataLoader(ct_test_dataset, batch_size=10, shuffle=True)
ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=10, shuffle=True)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    for X, y, fnames in ct_train_dataloader:
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
        pred = model(X)

        for idx, fname in enumerate(fnames):
            imgPatch = torch.reshape(pred[idx], (64, 64));
            fname_pred = fname.split("/")[-1]
            scio.savemat(ctc.HOME_DIR + "/data/patches/Predicted/" + fname_pred, {'imgPatch': np.array(imgPatch)})
