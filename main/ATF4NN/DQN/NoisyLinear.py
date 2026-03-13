import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinearDecay(nn.Module):
    """
    Noisy Linear Layer with Dynamic Noise Decay
    基于 Fortunato et al., 2017 的 NoisyNet 实现，
    加入 std_init 随训练步数衰减机制：
        std = std_init * (decay_rate ** decay_step)
    这样可以实现“前期探索强，后期稳定收敛”的效果。
    """
    def __init__(self, input_dim, output_dim, 
                 std_init=0.4, decay_rate=0.9998, min_std=0.2):
        """
        参数:
        ----------
        input_dim : 输入维度
        output_dim : 输出维度
        std_init : 初始噪声强度 (默认 0.4)
        decay_rate : 每步衰减系数 (默认 0.99)
        min_std : 最小噪声下限 (默认 0.1)
        """
        super(NoisyLinearDecay, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        self.decay_rate = decay_rate
        self.min_std = min_std
        self.decay_step = 0

        # 可学习参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))

        # 噪声变量
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self._update_sigma()

    def _update_sigma(self):
        """根据当前 step 动态调整噪声规模"""
        current_std = max(self.min_std, self.std_init * (self.decay_rate ** self.decay_step))
        self.weight_sigma.data.fill_(current_std / math.sqrt(self.input_dim))
        self.bias_sigma.data.fill_(current_std / math.sqrt(self.output_dim))

    def reset_noise(self):
        """重新采样噪声"""
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def step_decay(self):
        """在每个训练步骤或 episode 调用一次，用于更新衰减"""
        self.decay_step += 1
        self._update_sigma()
