import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import get_surrogate_names
from model import MLP


def train(surrogate='piecewise_exp', epochs=3, batch_size=128, lr=0.001, gamma=1.0, alpha=None,
          save_path='checkpoints/mnist_mlp.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Surrogate: {surrogate}, gamma: {gamma}, alpha: {alpha}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = MLP(surrogate=surrogate, gamma=gamma, alpha=alpha).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('开始训练...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%')

        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_accuracy = 100.0 * test_correct / len(test_dataset)
        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f'Epoch {epoch + 1} 完成: Train Loss: {avg_train_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2f}%\n')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存到 {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST SNN with selectable surrogate gradients.')
    parser.add_argument('--surrogate', default='piecewise_exp', choices=get_surrogate_names(),
                        help='替代梯度函数')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='兼容旧 ZIF 的宽度参数；alpha 未设置时 alpha=2/gamma')
    parser.add_argument('--alpha', type=float, default=None,
                        help='替代函数斜率/宽度参数；设置后优先于 gamma')
    parser.add_argument('--save-path', default='checkpoints/mnist_mlp.pth', help='模型保存路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        surrogate=args.surrogate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        save_path=args.save_path,
    )
