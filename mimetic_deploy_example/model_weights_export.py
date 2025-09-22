# export_weights.py
import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

input_size = 784  # 28x28像素
hidden_size = 128
num_classes = 10  # 0-9数字

model = SimpleNN(input_size, hidden_size, num_classes)

model.load_state_dict(torch.load('model.pt'))

# 导出权重到文本文件
def export_weights(model, filename):
    with open(filename, 'w') as f:
        for name, param in model.named_parameters():
            if 'weight' in name:
                f.write(f"{name}\n")
                np.savetxt(f, param.detach().numpy().flatten(), fmt='%.6f')
                f.write("\n")
            elif 'bias' in name:
                f.write(f"{name}\n")
                np.savetxt(f, param.detach().numpy().flatten(), fmt='%.6f')
                f.write("\n")

export_weights(model, 'model_weights.txt')