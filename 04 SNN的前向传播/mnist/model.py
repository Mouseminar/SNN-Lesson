import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


class MLP(nn.Module):
    """简单的多层感知机模型用于MNIST分类"""

    def __init__(self, input_size=784, hidden_size=512, num_classes=10, T=6):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc3_s = tdLayer(self.fc3)

        self.act = LIFSpike()
        self.T = T

    def lif_step(self, x_t, mem):
        """
        单个时间步的LIF更新：
        mem[t] = tau * mem[t-1] + x[t]
        spike[t] = H(mem[t] - thresh)
        mem[t] = (1 - spike[t]) * mem[t]
        """
        mem = mem * self.act.tau + x_t
        spike = self.act.act(mem - self.act.thresh, self.act.gamma)
        mem = (1.0 - spike) * mem
        return spike, mem

    def forward(self, x):
        # 将图像展平为一维向量 [B, 1, 28, 28] -> [B, 784]
        x = x.view(-1, 28 * 28)

        # 扩展时间维度 [B, 784] -> [B, T, 784]
        x = add_dimention(x, self.T)

        # ======================== 逐层传播（广度优先） ========================
        x = self.fc1_s(x)
        x = self.act(x)
        x = self.fc2_s(x)
        x = self.act(x)
        x = self.fc3_s(x)
        x = x.mean(1)
        return x

        # # ======================== 逐步传播（深度优先） ========================
        # batch_size = x.size(0)
        # device = x.device

        # mem1 = torch.zeros(batch_size, self.fc1.out_features, device=device, dtype=x.dtype)
        # mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device, dtype=x.dtype)

        # out_seq = []

        # for t in range(self.T):
        #     x_t = x[:, t, :]                 # [B, 784]

        #     h1 = self.fc1(x_t)               # 第1层线性
        #     s1, mem1 = self.lif_step(h1, mem1)

        #     h2 = self.fc2(s1)                # 第2层线性
        #     s2, mem2 = self.lif_step(h2, mem2)

        #     out_t = self.fc3(s2)             # 输出层不加脉冲
        #     out_seq.append(out_t)

        # x = torch.stack(out_seq, dim=1)      # [B, T, num_classes]
        # x = x.mean(1)                        # 时间维平均
        
        # return x