import collections
import torch
from .Network import Net
import numpy as np
from .MaxQueue import MaxTupleArray
from .Replay import ReplayBuffer
from ...graph import find_topo_sort_priority
import torch.nn.functional as F
import os
class Game:
    def __init__(self, length, model_name, max_fusing_node=10, h_feats=64, epoch=1, topk=10, replay_times=10):
        self.in_feats = length
        self.length = length
        self.replay_times = replay_times
        self.max_fusing_node = max_fusing_node
        self.model_name = model_name
        self.num_list = []
        for i in range(length):
            self.num_list.append(i)
        self.h_feats = h_feats
        self.step = 0
        self.max_step = length
        self.batch_size_pos = 256
        self.batch_size_neg = 256
        self.buffer_size = 32768
        self.gamma = 0.99  # 折扣因子
        self.n_step = 3
        # self.epoch = epoch
        self.epoch = epoch
        self.result = MaxTupleArray(topk)

        self.q_net = Net(self.in_feats, self.h_feats, self.max_fusing_node).cuda()  # 估计动作价值 神经网络
        self.target_q_net = Net(self.in_feats, self.h_feats, self.max_fusing_node).cuda()  # 计算目标值 神经网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=1e-5)

        self.epsilon = 0.9995
        self.epsilon_decay = 0.9995  # 衰减因子
        self.epsilon_min = 0.1 # 探索率最小值
        self.update_rate = 1000

        self.memory_pos = ReplayBuffer(int(self.buffer_size), kind=1)
        self.memory_neg = ReplayBuffer(int(self.buffer_size))
        self.reward_array = []
        self.action_array = []
        self.t_matrix = None

    def replay(self, t_memory, batch_size):
        # 从回放经验中抽取数据
        #print("t_memory:",len(t_memory)," capacity:",t_memory.capacity)
        if not(len(t_memory)==0):
            states, actions, next_states, rewards, idxs, is_weights = t_memory.sample(batch_size)
            #print("weights:",is_weights)
            # 计算当前Q值
            current_q_values = self.q_net(states)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # DDQN: 用在线网络选择动作，用目标网络评估价值
            with torch.no_grad():
                # 在线网络选择next_state的最佳动作
                next_q_values_online = self.q_net(next_states)
                next_actions = next_q_values_online.argmax(1)
        
                # 目标网络评估这些动作的价值
                next_q_values_target = self.target_q_net(next_states)
                next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
                # 计算目标Q值
                target_q = rewards + self.gamma * next_q
    
            # 计算TD误差（用于更新优先级）
            td_errors = torch.abs(current_q - target_q)
    
            # 计算加权loss
            loss = (is_weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
            #print("Loss:",loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t_memory.batch_update(idxs, td_errors, rewards)

            # 通过反向传播更新神经网络的参数
            self.step += 1
            # 隔一定步数 更新目标网络
            #print("current_step:",self.step," update rate:",self.update_rate)
            if self.step % self.update_rate == 0:
                #print("Update q net!")
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                self.step = 0

        return t_memory

    def train(self, ordered_nodes, node_to_index, engine):
        output_list = list(filter(lambda node : node.is_output(), ordered_nodes))
        ordered_nodes = find_topo_sort_priority(output_list)
        reward_list = []

        engine.node_topo_id = {ordered_nodes[i]: i for i in range(len(ordered_nodes))}
        for t_epoch in range(self.epoch):
            new_shape_vec = torch.zeros(self.in_feats).cuda()
            node_in_group = []
            node_index_in_group = []
            fg = []
            temp_num_list = self.num_list[:]
            sum_reward = 0
            engine.reset()

            for t_node in ordered_nodes:
                if t_node in node_in_group or t_node.is_output() or t_node.is_placeholder():
                    continue
                old_shape_vec = new_shape_vec.clone()
                _Q_tensor = self.q_net(old_shape_vec)
                Prob = torch.rand(1).cuda()
                if Prob > self.epsilon:
                    max_indices = torch.argmax(_Q_tensor)   # 获取最大值的索引
                    t_max_fusing_node_ori = int(max_indices)
                else:
                    t_max_fusing_node_ori = int(torch.round(torch.rand(1).cuda() * self.max_fusing_node))

                t_node_in_group = node_in_group[:]
                t_node_index_in_group = node_index_in_group[:]
                t_new_shape_vec = old_shape_vec.clone()
                t_temp_num_list = temp_num_list[:]

                if t_epoch != 0:
                    t_fg = engine._build_fusion_group(t_node, t_max_fusing_node_ori)
                    t_max_fusing_node = t_max_fusing_node_ori
                else:
                    t_fg = engine._build_fusion_group(t_node, self.max_fusing_node)
                    t_max_fusing_node = self.max_fusing_node
                #t_max_fusing_node = len(t_fg.nodes)

                for node in t_fg.nodes:
                    t_new_shape_vec[node_to_index[node]] += 1
                    t_node_in_group.append(node)
                    t_node_index_in_group.append(node_to_index[node])
                    t_temp_num_list.remove(node_to_index[node])

                print("--------------------------------------------------------------------------------")
                print("new_shape_vec:", t_new_shape_vec)
                print("Epoch:", t_epoch, " reward:", sum_reward, " t_max_node:", t_max_fusing_node, " epsilon:",self.epsilon)
                print("Top-K:",self.result.get_maximums_reward())
                print("--------------------------------------------------------------------------------")

                flag2 = torch.all(t_new_shape_vec == 1)
                if flag2:
                    t_reward = 10.0 + t_fg.gain
                else:
                    if t_fg.gain > 0:
                        t_reward = t_fg.gain
                    else:
                        t_reward = t_fg.gain - 1

                if t_fg.gain > 0:
                    sum_reward += t_fg.gain

                fg.append(t_fg)
                node_in_group = t_node_in_group[:]
                node_index_in_group = t_node_index_in_group[:]
                new_shape_vec = t_new_shape_vec.clone()
                temp_num_list = t_temp_num_list[:]

                #if t_epoch != 0:
                if t_reward > 0:
                    self.memory_pos.push(old_shape_vec, t_max_fusing_node, t_new_shape_vec, t_reward)
                else:
                    self.memory_neg.push(old_shape_vec, t_max_fusing_node, t_new_shape_vec, t_reward)
                
                for _ in range(self.replay_times):
                    self.replay(self.memory_pos, self.batch_size_pos)
                    self.replay(self.memory_neg, self.batch_size_neg)
                if flag2:
                    self.result.insert(fg, sum_reward)
                    break
            reward_list.append(sum_reward)
            self.q_net.reset_noise_all()
            self.q_net.step_decay_all()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        os.makedirs("./res/"+self.model_name+"/", exist_ok=True)
        torch.save(reward_list, "./res/"+self.model_name+"/reward_list.pt")
        results = self.result.get_maximums()
        return list(results[0])
