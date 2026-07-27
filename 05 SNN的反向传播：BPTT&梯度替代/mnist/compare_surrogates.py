import argparse
import csv
import os
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from layers import get_surrogate_names
from model import MLP


COLORS = {
    'atan': '#0072B2',
    'piecewise_exp': '#D55E00',
    'rectangular': '#009E73',
    'sigmoid': '#CC79A7',
    'triangular': '#E69F00',
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def limit_dataset(dataset, max_samples, seed=0):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one_run(name, seed, args, train_dataset, test_loader, device):
    set_seed(seed)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=args.num_workers,
    )
    model = MLP(T=args.timesteps, hidden_size=args.hidden_size,
                surrogate=name, gamma=args.gamma, alpha=args.alpha).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.perf_counter()
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss = train_loss_sum / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        row = {
            'surrogate': name,
            'seed': seed,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        history.append(row)
        print(f'[{name} seed={seed}] epoch {epoch + 1}/{args.epochs}: '
              f'train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%')

    elapsed = time.perf_counter() - start_time
    return history, elapsed


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def aggregate_histories(histories_by_surrogate, seconds_by_surrogate, target_acc):
    summaries = []
    curves = {}

    for name, runs in histories_by_surrogate.items():
        epochs = sorted({row['epoch'] for run in runs for row in run})
        curve = []
        for epoch in epochs:
            rows = [row for run in runs for row in run if row['epoch'] == epoch]
            curve.append({
                'surrogate': name,
                'epoch': epoch,
                'train_loss_mean': mean([row['train_loss'] for row in rows]),
                'train_loss_std': std([row['train_loss'] for row in rows]),
                'train_acc_mean': mean([row['train_acc'] for row in rows]),
                'train_acc_std': std([row['train_acc'] for row in rows]),
                'test_loss_mean': mean([row['test_loss'] for row in rows]),
                'test_loss_std': std([row['test_loss'] for row in rows]),
                'test_acc_mean': mean([row['test_acc'] for row in rows]),
                'test_acc_std': std([row['test_acc'] for row in rows]),
            })
        curves[name] = curve

        final = curve[-1]
        best_epoch = max(curve, key=lambda row: (row['test_acc_mean'], -row['test_loss_mean']))
        auc = mean([row['test_acc_mean'] for row in curve])
        reached = [row['epoch'] for row in curve if row['test_acc_mean'] >= target_acc]
        epoch_to_target = reached[0] if reached else None
        summaries.append({
            'surrogate': name,
            'runs': len(runs),
            'final_train_acc_mean': final['train_acc_mean'],
            'final_train_acc_std': final['train_acc_std'],
            'final_test_acc_mean': final['test_acc_mean'],
            'final_test_acc_std': final['test_acc_std'],
            'final_test_loss_mean': final['test_loss_mean'],
            'final_test_loss_std': final['test_loss_std'],
            'best_epoch': best_epoch['epoch'],
            'best_test_acc_mean': best_epoch['test_acc_mean'],
            'best_test_acc_std': best_epoch['test_acc_std'],
            'test_acc_auc': auc,
            'epoch_to_target': epoch_to_target,
            'seconds_mean': mean(seconds_by_surrogate[name]),
            'seconds_std': std(seconds_by_surrogate[name]),
        })

    summaries.sort(key=lambda row: (row['final_test_acc_mean'], row['test_acc_auc']), reverse=True)
    return summaries, curves


def format_value_with_std(value, spread, suffix=''):
    return f'{value:.2f}+/-{spread:.2f}{suffix}'


def format_summary_table(summaries):
    headers = [
        'surrogate', 'runs', 'final_test_acc', 'best_epoch', 'best_test_acc',
        'test_acc_auc', 'epoch_to_target', 'seconds/run'
    ]
    table = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in summaries:
        epoch_to_target = '-' if row['epoch_to_target'] is None else str(row['epoch_to_target'])
        table.append('| ' + ' | '.join([
            row['surrogate'],
            str(row['runs']),
            format_value_with_std(row['final_test_acc_mean'], row['final_test_acc_std'], '%'),
            str(row['best_epoch']),
            format_value_with_std(row['best_test_acc_mean'], row['best_test_acc_std'], '%'),
            f"{row['test_acc_auc']:.2f}",
            epoch_to_target,
            format_value_with_std(row['seconds_mean'], row['seconds_std'], 's'),
        ]) + ' |')
    return '\n'.join(table)


def write_csv(raw_rows, curves, summaries, raw_csv_path, curve_csv_path, summary_csv_path):
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    with open(raw_csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'surrogate', 'seed', 'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    with open(curve_csv_path, 'w', encoding='utf-8', newline='') as file:
        fieldnames = [
            'surrogate', 'epoch', 'train_loss_mean', 'train_loss_std', 'train_acc_mean', 'train_acc_std',
            'test_loss_mean', 'test_loss_std', 'test_acc_mean', 'test_acc_std'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name in curves:
            writer.writerows(curves[name])

    with open(summary_csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def plot_curves(curves, plot_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, curve in curves.items():
        epochs = [row['epoch'] for row in curve]
        test_mean = [row['test_acc_mean'] for row in curve]
        test_std = [row['test_acc_std'] for row in curve]
        train_mean = [row['train_acc_mean'] for row in curve]
        train_std = [row['train_acc_std'] for row in curve]
        color = COLORS.get(name, None)

        axes[0].plot(epochs, test_mean, marker='o', linewidth=2, label=name, color=color)
        axes[0].fill_between(
            epochs,
            [m - s for m, s in zip(test_mean, test_std)],
            [m + s for m, s in zip(test_mean, test_std)],
            alpha=0.14,
            color=color,
        )
        axes[1].plot(epochs, train_mean, marker='o', linewidth=2, label=name, color=color)
        axes[1].fill_between(
            epochs,
            [m - s for m, s in zip(train_mean, train_std)],
            [m + s for m, s in zip(train_mean, train_std)],
            alpha=0.10,
            color=color,
        )

    axes[0].set_title('MNIST SNN surrogate comparison: test accuracy')
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].set_title('Training accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train accuracy (%)')
    axes[1].grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def write_report(args, summaries, curves, report_path, plot_path, raw_csv_path, curve_csv_path, summary_csv_path):
    best_final = max(summaries, key=lambda row: (row['final_test_acc_mean'], -row['final_test_loss_mean']))
    best_speed = max(summaries, key=lambda row: row['test_acc_auc'])
    table = format_summary_table(summaries)
    train_size = args.max_train_samples if args.max_train_samples > 0 else 'full'
    test_size = args.max_test_samples if args.max_test_samples > 0 else 'full'

    lines = [
        '# MNIST SNN Surrogate Gradient Comparison',
        '',
        f'- Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'- Surrogates: {", ".join(args.surrogates)}',
        f'- Seeds: {", ".join(str(seed) for seed in args.seeds)}',
        f'- Epochs: {args.epochs}',
        f'- Batch size: {args.batch_size}',
        f'- Learning rate: {args.lr}',
        f'- T: {args.timesteps}',
        f'- Hidden size: {args.hidden_size}',
        f'- Gamma: {args.gamma}',
        f'- Alpha: {args.alpha}',
        f'- Train samples: {train_size}',
        f'- Test samples: {test_size}',
        f'- Target accuracy for convergence: {args.target_acc:.2f}%',
        '',
        '## Accuracy Curves',
        '',
        f'![Surrogate training curves]({os.path.basename(plot_path)})',
        '',
        '## Summary',
        '',
        table,
        '',
        '## Conclusion',
        '',
        f'- Highest final mean test accuracy: **{best_final["surrogate"]}** '
        f'({best_final["final_test_acc_mean"]:.2f}+/-{best_final["final_test_acc_std"]:.2f}%).',
        f'- Fastest overall convergence by mean test-accuracy AUC: **{best_speed["surrogate"]}** '
        f'(AUC={best_speed["test_acc_auc"]:.2f}).',
        '- A higher AUC means the curve stayed higher across training, which captures both speed and final quality.',
        '',
        '## Files',
        '',
        f'- Plot: `{plot_path}`',
        f'- Raw per-run CSV: `{raw_csv_path}`',
        f'- Aggregated curve CSV: `{curve_csv_path}`',
        f'- Summary CSV: `{summary_csv_path}`',
        '',
        '## Per-Epoch Mean Curves',
        '',
    ]

    for name in args.surrogates:
        lines.append(f'### {name}')
        lines.append('')
        lines.append('| epoch | train_acc | test_acc | train_loss | test_loss |')
        lines.append('| --- | --- | --- | --- | --- |')
        for row in curves[name]:
            lines.append(
                f'| {row["epoch"]} | '
                f'{row["train_acc_mean"]:.2f}+/-{row["train_acc_std"]:.2f}% | '
                f'{row["test_acc_mean"]:.2f}+/-{row["test_acc_std"]:.2f}% | '
                f'{row["train_loss_mean"]:.4f}+/-{row["train_loss_std"]:.4f} | '
                f'{row["test_loss_mean"]:.4f}+/-{row["test_loss_std"]:.4f} |'
            )
        lines.append('')

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare surrogate gradients on MNIST SNN training.')
    parser.add_argument('--surrogates', nargs='+', default=get_surrogate_names(), choices=get_surrogate_names(),
                        help='需要对比的替代函数列表')
    parser.add_argument('--epochs', type=int, default=10, help='每个替代函数训练轮数')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2024, 2025, 2026], help='重复实验随机种子')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='兼容旧 ZIF 的宽度参数；alpha 未设置时 alpha=2/gamma')
    parser.add_argument('--alpha', type=float, default=None,
                        help='替代函数斜率/宽度参数；设置后优先于 gamma')
    parser.add_argument('--timesteps', type=int, default=6, help='SNN 时间步 T')
    parser.add_argument('--hidden-size', type=int, default=512, help='MLP 隐藏层宽度')
    parser.add_argument('--max-train-samples', type=int, default=0,
                        help='快速实验时限制训练样本数；0 表示完整训练集')
    parser.add_argument('--max-test-samples', type=int, default=0,
                        help='快速实验时限制测试样本数；0 表示完整测试集')
    parser.add_argument('--subset-seed', type=int, default=0, help='抽样子集随机种子')
    parser.add_argument('--target-acc', type=float, default=97.0, help='报告收敛轮数时使用的目标测试准确率')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--data-root', default='data', help='MNIST 数据目录')
    parser.add_argument('--report-path', default='mnist/surrogate_comparison_report.md', help='Markdown 报告输出路径')
    parser.add_argument('--plot-path', default='mnist/surrogate_training_curves.png', help='训练曲线图片输出路径')
    parser.add_argument('--raw-csv-path', default='mnist/surrogate_runs_raw.csv', help='每次实验原始结果 CSV')
    parser.add_argument('--curve-csv-path', default='mnist/surrogate_curves.csv', help='聚合训练曲线 CSV')
    parser.add_argument('--summary-csv-path', default='mnist/surrogate_summary.csv', help='汇总结果 CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Seeds: {args.seeds}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    train_dataset = limit_dataset(train_dataset, args.max_train_samples, args.subset_seed)
    test_dataset = limit_dataset(test_dataset, args.max_test_samples, args.subset_seed)
    test_loader = make_loader(test_dataset, batch_size=1000, shuffle=False, seed=args.subset_seed,
                              num_workers=args.num_workers)

    histories_by_surrogate = {name: [] for name in args.surrogates}
    seconds_by_surrogate = {name: [] for name in args.surrogates}
    raw_rows = []

    for name in args.surrogates:
        print(f'\n=== Surrogate: {name} ===')
        for seed in args.seeds:
            print(f'--- Run seed={seed} ---')
            history, elapsed = train_one_run(name, seed, args, train_dataset, test_loader, device)
            histories_by_surrogate[name].append(history)
            seconds_by_surrogate[name].append(elapsed)
            raw_rows.extend(history)

    summaries, curves = aggregate_histories(histories_by_surrogate, seconds_by_surrogate, args.target_acc)
    table = format_summary_table(summaries)
    print('\nSummary:')
    print(table)
    best_final = summaries[0]
    best_speed = max(summaries, key=lambda row: row['test_acc_auc'])
    print(f"\nBest final accuracy: {best_final['surrogate']} "
          f"({best_final['final_test_acc_mean']:.2f}+/-{best_final['final_test_acc_std']:.2f}%)")
    print(f"Fastest by AUC: {best_speed['surrogate']} (AUC={best_speed['test_acc_auc']:.2f})")

    plot_curves(curves, args.plot_path)
    write_csv(raw_rows, curves, summaries, args.raw_csv_path, args.curve_csv_path, args.summary_csv_path)
    write_report(
        args,
        summaries,
        curves,
        args.report_path,
        args.plot_path,
        args.raw_csv_path,
        args.curve_csv_path,
        args.summary_csv_path,
    )
    print(f'Plot written to: {args.plot_path}')
    print(f'Report written to: {args.report_path}')


if __name__ == '__main__':
    main()