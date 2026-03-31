import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MLP


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = MLP().to(device)
    ckpt_path = 'checkpoints/mnist_mlp.pth'

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    test_loss /= total
    test_accuracy = 100.0 * correct / total

    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    test()
