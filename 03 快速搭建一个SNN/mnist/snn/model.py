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

        
    def forward(self, x):
        # 将图像展平为一维向量
        x = x.view(-1, 28 * 28)
        
        x = add_dimention(x, self.T)
        
        x = self.fc1_s(x)
        x = self.act(x)
        
        x = self.fc2_s(x)
        x = self.act(x)

        # 输出层
        x = self.fc3_s(x)
        
        x = x.mean(1)
        
        return x