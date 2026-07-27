import torch


def main():
    x = torch.randn(8, requires_grad=True)
    spike = (x > 0).float()
    loss = spike.mean()

    print('x.requires_grad =', x.requires_grad)
    print('spike.requires_grad =', spike.requires_grad)
    print('loss.requires_grad =', loss.requires_grad)
    print('Calling loss.backward() now. This should raise RuntimeError because hard threshold has no grad_fn.')

    loss.backward()


if __name__ == '__main__':
    main()