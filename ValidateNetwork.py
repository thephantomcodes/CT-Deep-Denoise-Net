import numpy as np
from scipy import io as scio
import ctconfig as ctc
import torch
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import time
import os

s = 99
epochs = range(s, 100)
trial = 17
for epoch in epochs:
    dest = f"validationdata/{trial}/{epoch}/"
    model_src = f"modelstates/{trial}/model_{epoch}.pth"

    if os.path.isfile(model_src) is False:
        print(f"{model_src} not found")
        exit(0)

    if os.path.isdir(dest) is False:
        os.mkdir(dest)
    model = ctu.NeuralNetwork()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)
    model.load_state_dict(torch.load(model_src, map_location=torch.device(device)))
    if device == "cuda":
        model.cuda()

    # print(model)
    print(f"Epoch: {epoch}")

    ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/ellipsoids/Validation/")
    ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)

    model.eval()

    with torch.no_grad():
        for bat, (image, label, file_names) in enumerate(ct_valid_dataloader):
            print(f"Batch {bat} - {time.asctime(time.localtime(time.time()))}")
            image, label = torch.tensor(image).to(device), torch.tensor(label).to(device)
            pred = model(image)

            for idx, fname in enumerate(file_names):
                imgPatch = pred[idx][0].cpu()
                # print(torch.max(imgPatch) - torch.min(imgPatch))
                fname_pred = fname.split("/")[-1]
                scio.savemat(f"{dest}/{fname_pred}", {'imgPatch': np.array(imgPatch)})
