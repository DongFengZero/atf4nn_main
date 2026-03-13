import torch
import torch.nn as nn
import torch
from .NoisyLinear import NoisyLinearDecay

class Net(nn.Module):
    def __init__(self, in_feats, hidden_feats, max_fusing_nodes):
        super(Net, self).__init__()
        self.max_fusing_nodes = max_fusing_nodes + 1

        # 使用动态衰减版本的 NoisyLinearDecay
        self.noisy_linear = NoisyLinearDecay(in_feats, hidden_feats, std_init=0.4)
        self.noisy_linear2 = NoisyLinearDecay(hidden_feats, hidden_feats, std_init=0.4)
        self.noisy_linear3 = NoisyLinearDecay(hidden_feats, hidden_feats, std_init=0.4)
        self.noisy_linear4 = NoisyLinearDecay(hidden_feats, hidden_feats, std_init=0.4)
        self.noisy_linear5 = NoisyLinearDecay(hidden_feats, self.max_fusing_nodes, std_init=0.4)

    def forward(self, shape_vec):
        hidden_vec = self.noisy_linear(shape_vec)
        hidden_vec = torch.tanh(self.noisy_linear2(hidden_vec))
        hidden_vec = torch.tanh(self.noisy_linear3(hidden_vec))
        hidden_vec1 = torch.tanh(self.noisy_linear4(hidden_vec))
        Q_value_vec = self.noisy_linear5(hidden_vec1)
        return Q_value_vec

    def reset_noise_all(self):
        """重采样全部 Noisy 层的噪声"""
        for m in self.modules():
            if isinstance(m, NoisyLinearDecay):
                m.reset_noise()

    def step_decay_all(self):
        """统一衰减全部 Noisy 层的噪声强度"""
        for m in self.modules():
            if isinstance(m, NoisyLinearDecay):
                m.step_decay()
