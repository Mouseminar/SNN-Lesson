import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
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


def make_loaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    if args.train_samples > 0:
        train_dataset = Subset(train_dataset, list(range(args.train_samples)))
    if args.test_samples > 0:
        test_dataset = Subset(test_dataset, list(range(args.test_samples)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return loss_sum / total, 100.0 * correct / total


def train_model(label, model, train_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        rows.append({
            'model': label,
            'epoch': epoch + 1,
            'train_acc': 100.0 * correct / total,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'fc1_grad': grad_norm(model.fc1.weight),
            'fc2_grad': grad_norm(model.fc2.weight),
            'fc3_grad': grad_norm(model.fc3.weight),
        })
    return rows


def print_rows(rows):
    print('| model | epoch | train_acc | test_acc | test_loss | fc1_grad | fc2_grad | fc3_grad |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in rows:
        print('| {model} | {epoch} | {train_acc:.2f}% | {test_acc:.2f}% | {test_loss:.4f} | {fc1_grad} | {fc2_grad} | {fc3_grad} |'.format(**row))


def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST SNN with and without surrogate gradient.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--test-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data-root', default=os.path.join(ROOT_DIR, 'data'))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    set_seed(args.seed)
    train_loader, test_loader = make_loaders(args)

    set_seed(args.seed)
    surrogate_model = MLP(surrogate='triangular').to(device)
    set_seed(args.seed)
    no_surrogate_model = NoSurrogateMLP().to(device)

    rows = []
    rows.extend(train_model('surrogate_triangular', surrogate_model, train_loader, test_loader, args, device))
    rows.extend(train_model('no_surrogate', no_surrogate_model, train_loader, test_loader, args, device))
    print_rows(rows)


if __name__ == '__main__':
    main()