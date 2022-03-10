import torch
import numpy as np
import random as rd
from PIL import Image
from torchvision.transforms import RandomRotation
from torchvision.transforms import Compose

from ulgie import resize1, pad, totensor, resize2, C8SteerableCNN, MnistRotDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(0)
rd.seed(0)
torch.random.manual_seed(0)

eta = C8SteerableCNN().to(device)
mu = C8SteerableCNN().to(device)
delta = C8SteerableCNN().to(device)

# build the test set
train_file = "../mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat" # train
test_file = "../mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat" # test

train_transform = Compose([
    pad,
    resize1,
    RandomRotation(180, resample=Image.BILINEAR, expand=False),
    resize2,
    totensor,
])

mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)

loss = torch.nn.MSELoss()
params = list(eta.parameters()) + list(mu.parameters()) + list(delta.parameters())
optimizer = torch.optim.Adam(params, lr=5e-5, weight_decay=1e-5)

for epoch in range(31):
    eta.train()
    mu.train()
    delta.train()
    for i, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)

        y = delta(eta(x))

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):
                x = x.to(device)
                t = t.to(device)

                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct / total * 100.}")