import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_DIR = os.path.join(ROOT_DIR, 'mnist')
if MNIST_DIR not in sys.path:
    sys.path.insert(0, MNIST_DIR)

from layers import add_dimention, tdLayer  # noqa: E402
from model import MLP  # noqa: E402


class HardLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5):
        super().__init__()
        self.thresh = thresh
        self.tau = tau

    def forward(self, x):
        mem = 0
        spikes = []
        for t in range(x.shape[1]):
            mem = mem * self.tau + x[:, t, ...]
            spike = (mem - self.thresh > 0).float()
            mem = (1.0 - spike) * mem
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class NoSurrogateMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, T=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc3_s = tdLayer(self.fc3)
        self.act = HardLIFSpike()
        self.T = T

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = add_dimention(x, self.T)
        x = self.fc1_s(x)
        x = self.act(x)
        x = self.fc2_s(x)
        x = self.act(x)
        x = self.fc3_s(x)
        return x.mean(1)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad_norm(parameter):
    if parameter.grad is None:
        return 'None'
    return f'{parameter.grad.norm().item():.6e}'


def make_loader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    if args.train_samples > 0:
        train_dataset = Subset(train_dataset, list(range(args.train_samples)))
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


def run_one_batch(name, model, data, target, criterion):
    model.zero_grad(set_to_none=True)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    print('| {} | {:.4f} | {} | {} | {} |'.format(
        name,
        loss.item(),
        grad_norm(model.fc1.weight),
        grad_norm(model.fc2.weight),
        grad_norm(model.fc3.weight),
    ))


def parse_args():
    parser = argparse.ArgumentParser(description='Probe gradients with and without surrogate gradient.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data-root', default=os.path.join(ROOT_DIR, 'data'))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(args)
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    criterion = nn.CrossEntropyLoss()

    surrogate_model = MLP(surrogate='triangular').to(device)
    no_surrogate_model = NoSurrogateMLP().to(device)

    print('Using device:', device)
    print('| model | loss | fc1.weight.grad | fc2.weight.grad | fc3.weight.grad |')
    print('| --- | --- | --- | --- | --- |')
    run_one_batch('surrogate_triangular', surrogate_model, data, target, criterion)
    run_one_batch('no_surrogate', no_surrogate_model, data, target, criterion)


if __name__ == '__main__':
    main()