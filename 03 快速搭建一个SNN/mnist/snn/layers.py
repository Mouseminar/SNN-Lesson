import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================== 张量归一化模块 ===========================

class TensorNormalization(nn.Module):
    """
    自定义归一化模块，用于将输入张量按通道进行均值/方差标准化
    """
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def forward(self, X):
        return normalizex(X, self.mean, self.std)

def normalizex(tensor, mean, std):
    """
    实际的归一化操作，支持跨设备
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

# =========================== 时间序列输入容器 ===========================

class SeqToANNContainer(nn.Module):
    """
    用于将带时间维度的输入 [B, T, ...] 展平为 [B*T, ...] 再送入ANN层中计算，
    然后再reshape回原来的时间结构 [B, T, ...]。
    """
    def __init__(self, *args):  # 可变参数列表
        super().__init__()
        if len(args) == 1:
            self.module = args[0]   # 只有一个模块时直接使用
        else:
            self.module = nn.Sequential(*args)  # 多个模块时使用 nn.Sequential 组合

    def forward(self, x_seq: torch.Tensor): # x_seq: [B, T, ...]
        y_shape = [x_seq.shape[0], x_seq.shape[1]]  # 记录原始的 batch 和时间维度
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())   # 展平为 [B*T, ...]
        y_shape.extend(y_seq.shape[1:]) # 记录输出的形状
        return y_seq.view(y_shape)  # 重新 reshape 回原来的形状

# =========================== 脉冲网络卷积模块 ===========================

class Layer(nn.Module):
    """
    一个时序卷积层 + BN + LIF 激活
    """
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        return x

# =========================== 脉冲网络池化模块 ===========================

class APLayer(nn.Module):
    """
    平均池化 + LIF 激活，用于时序池化处理
    """
    def __init__(self, kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        return x

# =========================== 梯度替代函数 ===========================

class ZIF(torch.autograd.Function):
    """
    Zero-In One-Out Fire 函数的定义，用于脉冲神经元的近似激活。
    前向传播输出脉冲，反向传播时使用软梯度替代，保持可微。
    staticmethod是为了让函数可以在没有实例化对象的情况下调用。
    """
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()   # 触发脉冲
        L = torch.tensor([gamma])    # 斜率参数
        ctx.save_for_backward(input, out, L)    # 保存输入和输出
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, out, others = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) ** 2 * ((gamma - input.abs()).clamp(min=0)) # clamp(min=0) 确保不小于0
        grad_input = grad_input * tmp
        return grad_input, None

# =========================== LIF 激活函数 ===========================

class LIFSpike(nn.Module):
    """
    LIF 神经元模型：
    - 使用指数衰减整合输入
    - 超过阈值就放电（产生spike）
    - 放电后膜电位重置
    """
    def __init__(self, thresh=1.0, tau=0.5, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh  # 放电阈值
        self.tau = tau        # 衰减系数
        self.gamma = gamma      # 梯度替代曲线斜率

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]  # 积分
            spike = self.act(mem - self.thresh, self.gamma)  # 触发放电
            mem = (1 - spike) * mem  # 放电后的膜电位重置
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)  # 按时间维度堆叠脉冲

# =========================== 时间维度添加工具函数 ===========================

def add_dimention(x, T):
    """
    将 [B, feature] 变成 [B, T, feature]，复制 T 次表示每个时间步一样的输入
    """
    x.unsqueeze_(1)         # 在时间维度上添加一个维度
    x = x.repeat(1, T, 1)   # 在时间维度上复制 T 次
    return x

# =========================== 时间维度 ANN 封装层 ===========================

class tdLayer(nn.Module):
    """
    将普通线性层/卷积层封装为支持时间维度的形式 [B, T, ...]
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

# =========================== 时间维度 BN 封装层（2D） ===========================

class tdBatchNorm(nn.Module):
    """
    支持时间序列输入的 BatchNorm2d 操作
    """
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        return self.seqbn(x)
