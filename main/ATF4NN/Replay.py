import random
import torch
import torch.nn.functional as F
import numpy as np
from .SumTree import SumTree, MaxSumTree

class ReplayBuffer:
    def __init__(self, capacity, kind=0, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 1e-4
        self.abs_err_upper = 1.0
	self.abs_rewards = [None] * capacity
	self.min_abs_reward = float('inf')
        
        if kind == 1:
            self.buffer = MaxSumTree(self.capacity)
        else:
            self.buffer = SumTree(self.capacity)

    def push(self, state, action, next_state, reward):
        """
        添加经验，保留reward绝对值最大的top-K经验
        """
        abs_reward = abs(float(reward))

        # buffer未满：直接添加
        if self.buffer.n_entries < self.capacity:
            priority = abs_reward if abs_reward > 0 else self.epsilon
            data_idx = self.buffer.add(priority, (state, action, next_state, reward))

            # 更新追踪变量
            self.abs_rewards[data_idx] = abs_reward
            if abs_reward < self.min_abs_reward:
                self.min_abs_reward = abs_reward

            return True

        # buffer已满：检查是否大于最小值
        if abs_reward <= self.min_abs_reward:
            return False

        # 【优化】找到最小reward的位置（结合索引追踪）
        min_idx = self._find_min_reward_index()

        if min_idx == -1:
            return False

        old_abs_reward = self.abs_rewards[min_idx]

        # 【关键】使用replace_at在指定位置替换
        priority = abs_reward
        self.buffer.replace_at(min_idx, priority, (state, action, next_state, reward))

        # 更新追踪变量
        self.abs_rewards[min_idx] = abs_reward

        # 更新min_abs_reward
        if old_abs_reward == self.min_abs_reward:
            self._update_min_abs_reward()
        elif abs_reward < self.min_abs_reward:
            self.min_abs_reward = abs_reward

        return True

    def _find_min_reward_index(self):
        """找到最小reward绝对值的索引"""
        min_idx = -1
        min_val = float('inf')

        for i in range(self.buffer.n_entries):
            if self.abs_rewards[i] is not None and self.abs_rewards[i] < min_val:
                min_val = self.abs_rewards[i]
                min_idx = i

        return min_idx
    
    def sample(self, batch_size, device='cuda'):
        """采样batch"""
        if self.buffer.n_entries == 0:
            return None
        
        batch_size = min(batch_size, self.buffer.n_entries)
        pri_segment = self.buffer.total() / batch_size
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        priorities = []
        states = []
        actions = []
        next_states = []
        rewards_array = []
        idxs = []
        
        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, p, data = self.buffer.get_leaf(s)
            
            priorities.append(p)
            states.append(data[0])
            actions.append(data[1])
            next_states.append(data[2])
            rewards_array.append(data[3])
            idxs.append(idx)
        
        # 计算重要性采样权重
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.buffer.total()
        is_weights = np.power(self.buffer.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # 转换为tensor
        try:
            states = torch.stack(states).to(device)
            next_states = torch.stack(next_states).to(device)
        except:
            # 如果states不是tensor，尝试转换
            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_array = torch.tensor(rewards_array, dtype=torch.float32, device=device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=device)
        
        return states, actions, next_states, rewards_array, idxs, is_weights
    
    def batch_update(self, tree_idxs, abs_errors):
        """批量更新优先级"""
        abs_errors = abs_errors.detach().cpu().numpy() if isinstance(abs_errors, torch.Tensor) else abs_errors
        abs_errors += self.epsilon
        
        # 裁剪到上限
        abs_errors = np.minimum(abs_errors, self.abs_err_upper)
        
        # 转换为优先级
        ps = np.power(abs_errors, self.alpha)
        
        for ti, p in zip(tree_idxs, ps):
            self.buffer.update(ti, float(p))
    
    def __len__(self):
        return self.buffer.len()