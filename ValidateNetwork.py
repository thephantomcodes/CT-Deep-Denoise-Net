import numpy as np
import skimage.metrics
from scipy import io as scio
import ctconfig as ctc
import torch
from torch.utils.data import DataLoader
import CtImageDataSet as ctid
import CtImageUtils as ctu
import time
import os
from matplotlib import pyplot as plt
from skimage import metrics

epochs = [400]
# epochs = [200, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
trial = 3

show_all_img = True
img_to_show = ['ID_0016_AGE_0063_CONTRAST_1_CT_0_0.mat', 'ID_0047_AGE_0069_CONTRAST_1_CT_0_1.mat']
img_generate_baseline = False

plt.figure(figsize=(4.5, 4.5))
plt.xticks([])
plt.yticks([])

for epoch in epochs:
    dest = f"validationdata/fun_{trial}/{epoch}/"
    # dest_plot = f"plots/fun_{trial}/{epoch}/"
    dest_plot = f"plots/fun_{trial}/"
    model_src = f"modelstates/fun_{trial}/model_{epoch}.pth"
    ct_valid_dataset = ctid.CtImageDataset(ctc.HOME_DIR + "/data/mat_norm_fbp_quads/Validation/")
    ct_valid_dataloader = DataLoader(ct_valid_dataset, batch_size=ctc.BATCH_SIZE, shuffle=True)

    if os.path.isfile(model_src) is False:
        print(f"{model_src} not found")
        exit(0)
    if os.path.isdir(dest) is False:
        os.mkdir(dest)
    if os.path.isdir(dest_plot) is False:
        os.mkdir(dest_plot)

    model = ctu.NeuralNetwork()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    model.load_state_dict(torch.load(model_src, map_location=torch.device(device)))
    if device == "cuda":
        model.cuda()

    # print(model)
    print(f"Epoch: {epoch}")

    model.eval()

    with torch.no_grad():
        for bat, (image, label, file_names) in enumerate(ct_valid_dataloader):
            # print(f"Batch {bat} - {time.asctime(time.localtime(time.time()))}")
            image, label = torch.tensor(image).to(device), torch.tensor(label).to(device)
            pred = model(image)

            for idx, fname in enumerate(file_names):
                file_name = fname.split('/')[-1]

                if show_all_img or file_name in img_to_show:
                    imgPatch = image[idx][0].cpu()
                    predPatch = pred[idx][0].cpu()
                    lblPatch = label[idx][0].cpu()
                    label_psnr = metrics.peak_signal_noise_ratio(imgPatch.numpy(), lblPatch.numpy())
                    label_ssim = metrics.structural_similarity(imgPatch.numpy(), lblPatch.numpy())
                    pred_psnr = metrics.peak_signal_noise_ratio(imgPatch.numpy(), predPatch.numpy())
                    pred_ssim = metrics.structural_similarity(imgPatch.numpy(), predPatch.numpy())

                    # show original and corrupted version, don't need to redo it
                    if img_generate_baseline:
                        plt.title('FBP\nAll Projections')
                        plt.imshow(lblPatch, cmap="gray")
                        plt.tight_layout()
                        plt.savefig(f"{dest_plot}/{file_name.replace('.mat', f'_0_full.png')}")

                        plt.title(f"FBP 1/4 Projections\nPSNR: {label_psnr:.2f} SSIM {label_ssim:.4f}")
                        plt.imshow(imgPatch, cmap="gray")
                        plt.tight_layout()
                        plt.savefig(f"{dest_plot}/{file_name.replace('.mat', f'_0_quarter.png')}")

                    plt.title(f"Denoised - Epoch {epoch}\nPSNR: {pred_psnr:.2f} SSIM {pred_ssim:.4f}")
                    plt.imshow(imgPatch - predPatch, cmap="gray")
                    plt.tight_layout()
                    plt.savefig(f"{dest_plot}/{file_name.replace('.mat', f'_{epoch}_denoised.png')}")

                    plt.title(f"Noise - Epoch {epoch}\n")
                    plt.imshow(predPatch, cmap="gray")
                    plt.tight_layout()
                    plt.savefig(f"{dest_plot}/{file_name.replace('.mat', f'_{epoch}_noise.png')}")

                    print(epoch, file_name, pred_psnr, pred_ssim, label_psnr, label_ssim)
                    # print(torch.max(imgPatch) - torch.min(imgPatch))
                    scio.savemat(f"{dest}/{file_name}", {'imgPatch': np.array(predPatch)})
