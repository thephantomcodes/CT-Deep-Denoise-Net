import torch

# create an 3 D tensor with 8 elements each
a = torch.randn([1,2,3,2,2])
# a = torch.tensor([[[[[ 0.7189, -0.8440],
#            [ 1.8482,  0.9630]],
#
#           [[ 0.0464,  0.9350],
#            [ 0.5001, -0.2602]]],
#
#
#          [[[ 0.5660,  0.1292],
#            [-0.1147, -0.1091]],
#
#           [[ 1.7358, -1.0722],
#            [ 0.8410, -0.2168]]]]])

# display actual  tensor
# b = a[0]
print(a)
b = a[:, :, 0]
print(b.shape)
print(b)
b = torch.reshape(b,(1,2,2,2))
print(b.shape)
# print(b)
