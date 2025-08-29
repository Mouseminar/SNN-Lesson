import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """简单的多层感知机模型用于MNIST分类"""
    
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 将图像展平为一维向量
        x = x.view(-1, 28 * 28)
        
        # 第一层：线性变换 + ReLU + Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层：线性变换 + ReLU + Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc3(x)
        return x