import numpy as np
from scipy import io as scio
import ctconfig as ctc
import torch
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import time

model = ctu.NeuralNetwork()
model.load_state_dict(torch.load("modelstates5/model_19.pth"))

# ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Training/")
# ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Test/")
# ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/patches/Validation/")

ct_train_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/noisyPatches/Training32/")
ct_test_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/noisyPatches/Test32/")
ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/noisy/Validation/")

ct_train_dataloader = DataLoader(ct_train_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)
ct_test_dataloader = DataLoader(ct_test_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)
ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    for bat, (image, label, file_names) in enumerate(ct_valid_dataloader):
        print(f"Batch {bat} - {time.asctime(time.localtime(time.time()))}")
        image, label = torch.tensor(image).to(device), torch.tensor(label).to(device)
        pred = model(image)

        for idx, fname in enumerate(file_names):
            imgPatch = pred[idx][0]
            # print(torch.max(imgPatch) - torch.min(imgPatch))
            fname_pred = fname.split("/")[-1]
            scio.savemat("./modelstates5/Predicted_19/" + fname_pred, {'imgPatch': np.array(imgPatch)})
