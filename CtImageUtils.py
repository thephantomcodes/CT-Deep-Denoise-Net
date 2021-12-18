import torch
from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64, 64*64),
            nn.ReLU(),
            nn.BatchNorm1d(64*64),
            nn.Linear(64*64, 64*64),
            nn.ReLU(),
            nn.BatchNorm1d(64*64),
            nn.Linear(64*64, 64*64)
        )

    def forward(self, x):
        x = self.flatten(x)
        # x = torch.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, torch.reshape(y, (10, 4096)))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, torch.reshape(y, (10, 4096))).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

