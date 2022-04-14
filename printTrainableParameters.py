import CtImageUtils as ctu

model = ctu.NeuralNetwork()
params = 0

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, parameter.numel())
        params = params + parameter.numel()

print(params)
