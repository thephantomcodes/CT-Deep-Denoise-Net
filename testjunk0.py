import torch
from pytorch_wavelets import DWTForward, DWTInverse
from matplotlib import pyplot as plt
import pywt
import pywt.data

cam_man = pywt.data.camera()
print(cam_man.shape)
C = torch.zeros(1,1,512,512)
C[0][0] = torch.tensor(cam_man)
print("C", C.shape)

xfm = DWTForward(wave='db1')  # Accepts all wave types available to PyWavelets
ifm = DWTInverse(wave='db1')

X = torch.randn(1,1,512,512)
C = X
Yl, Yh = xfm(C)
print("Yl", Yl.shape)
print("Yh", Yh[0].shape)

a = Yh[:][0][0]
print("a", a.shape)



ax = plt.subplot(141)
ax.set_title("Approx")
plt.imshow(Yl[0][0], cmap="gray")
ax = plt.subplot(142)
ax.set_title("Horiz")
plt.imshow(Yh[0][0][0][0], cmap="gray")
ax = plt.subplot(143)
ax.set_title("Vert")
plt.imshow(Yh[0][0][0][1], cmap="gray")
ax = plt.subplot(144)
ax.set_title("Diag")
plt.imshow(Yh[0][0][0][2], cmap="gray")
# plt.show()

print("Yl", Yl.shape)
for idx, yh in enumerate(Yh):
    print(f"Yh[{idx}]", yh.shape[2])

Y = ifm((Yl, Yh))
print(Y.shape)
print(torch.max(torch.abs(C-Y)))
print(torch.max(C))
print(torch.max(Y))

# plt.imshow(Y[0][0], cmap="gray")
# plt.show()