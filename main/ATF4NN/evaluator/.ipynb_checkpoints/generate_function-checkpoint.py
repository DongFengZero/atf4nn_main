from .cache import FileCache
from ...policy.common import coalesced_tensor_shape, coalesced_tensor_shape4
import numpy as np
import tvm

import re
import torch
from sortedcontainers import SortedDict
from collections import defaultdict
from ...te_utils import *
class TimeSeriesMaxStore:
    def __init__(self):
        self.data = defaultdict(SortedDict)  # data[name][timestep] = value

    def insert(self, name, timestep, value):
        self.data[name][timestep] = value

    def query(self, name, timestep):
        if name not in self.data:
            return 0
        ts_dict = self.data[name]
        idx = ts_dict.bisect_right(timestep)  # 找到第一个 t > timestep 的位置
        if idx == len(ts_dict):
            return 0
        # 从 idx 开始遍历后续项，返回最大值
        return max(v for t, v in list(ts_dict.items())[idx:])

class PartitionCompute:
    def __init__(self, td, tile, capacity, nodes, arch, shape_map2):
        self.cache = FileCache(capacity)
        self.shape_map2 = shape_map2
        self.arch = arch
        self.tile = tile
        self.td = td
        self.ori_capacity = capacity
        self.capacity = capacity
        self.nodes = nodes
        self.X_values = 0
        self.sub_computation_size = 0
        self.sub_computation_size2 = 0
        self.shape_map = None
        self.input_size = 0
        self.output_size = 0

    def update_capacity_origin(self, factor):
        self.capacity = self.ori_capacity/factor
        self.cache = FileCache(self.capacity)
    def get_node_compute_factor(self, node_name: str, default_factor: float = 1.0) -> float:
        # 定义计算因子字典
        compute_factor = {
            'Convolution': 18.0,  # 3x3核，输入平均影响多个输出
            'DepthwiseConv2dNative': 9.0,  # 每输入仅参与对应通道卷积的9次乘加
            'BatchMatMul': 2.0,  # 输入参与一次乘加（典型）
            'Dot': 2.0,
            'DotSplitK': 2.0,
            'Power': 1.0,
            'Sqrt': 1.0,
            'Square': 1.0,
            'SoftmaxBasic': 2.0,  # exp + reduce
            'Sigmoid': 1.5,
            'Tanh': 2.0,
            'Erf': 2.0,
            'Multiply': 1.0,
            'Divide': 1.0,
            'Add': 1.0,
            'Subtract': 1.0,
            'Maximum': 1.0,
            'Minimum': 1.0,
            'AvgPool': 1.0,  # 每输入参与一个池化窗口输出
            'MaxPool': 1.0,
            'Relu': 1.0,
            'HardSigmoid': 2.0,
            'DepthToSpace': 0,
            'Concat': 0,
            'Broadcast': 0,
            'Reshape': 0,
            'Slice': 0,
            'GatherV2': 0,
            'Resize': 2.0,
            'Convert': 0.2,
            'Sum': 1.0
        }

        # 使用下划线分割并去掉纯数字
        tokens = [tok for tok in node_name.split("_") if not tok.isdigit()]

        # 累加所有匹配的算子因子
        total_factor = 0.0
        for token in tokens:
            total_factor += compute_factor.get(token, default_factor)

        return total_factor + default_factor

    def update_map(self, shape_map):
        self.shape_map = shape_map
        self.sub_computation_size = 0
        self.input_size = 0
        self.output_size = 0
        for node in self.nodes:
            temp_list_input_index = []
            temp_list_output_index = []
            node_factor = self.get_node_compute_factor(node.name)
            sub_computation_size = 0
            for t_node_op in node.compute_ops:
                # input
                for t_source_node_op in t_node_op.input_tensors:
                    node_name = node.name
                    t_sub_size = self.calculate_input_tensor_size(node, t_source_node_op, self.shape_map, 2)
                    sub_computation_size += self.calculate_input_tensor_size(node, t_source_node_op, self.shape_map, 0) * node_factor
                    if t_source_node_op.name[:5] == "input":
                        index = int(t_source_node_op.name[5:])
                        if index not in temp_list_input_index:
                            temp_list_input_index.append(index)
                            if index < len(node.inputs) and not node.inputs[index].src_node.is_placeholder():
                                self.input_size += t_sub_size
                        if index < len(node.inputs):
                            src_node = node.inputs[index].src_node
                            if src_node in self.nodes:
                                continue
                    if not self.cache.find_file_size(node_name + "_" + t_source_node_op.name):
                        self.cache.set_file_size(node_name + "_" + t_source_node_op.name, t_sub_size)
                # output
                for i in range(t_node_op.num_outputs):
                    node_name = node.name
                    t_node_op_output = t_node_op.output(i)
                    t_sub_size = self.calculate_output_tensor_size(node, t_node_op_output, self.shape_map, 2)
                    # sub_computation_size += self.calculate_output_tensor_size(node, t_node_op_output, self.shape_map, 0)* node_factor
                    if t_node_op_output.name[:6] == "output" and int(t_node_op_output.name[6:]) not in temp_list_output_index:
                        temp_list_output_index.append(int(t_node_op_output.name[6:]))
                        if not node.outputs[int(t_node_op_output.name[6:])].dst_node.is_output():
                            self.output_size += t_sub_size
                    self.cache.set_file_size(node_name + "_" + t_node_op_output.name, t_sub_size)
            self.sub_computation_size += sub_computation_size

    def calculate_input_tensor_size(self, node, t_node_op, shape_map, option=0):
        op_name = t_node_op.name
        name = node.name
        if option == 1:
            read_transaction_elements = self.arch.transaction_size[1] // ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
            traffic = coalesced_tensor_shape4(shape_map[name][op_name], self.shape_map2[name][op_name], read_transaction_elements) * ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
        else:
            traffic = np.prod(shape_map[name][op_name])
            if option == 2:
                traffic = traffic * ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
        return traffic

    def calculate_output_tensor_size(self, node, t_node_op, shape_map, option=0):
        name = t_node_op.name
        if option == 1:
            write_transaction_elements = self.arch.transaction_size[0] // ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
            traffic = coalesced_tensor_shape4(shape_map[node.name][name], self.shape_map2[node.name][name], write_transaction_elements) * ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
        else:
            traffic = np.prod(shape_map[node.name][name])
            if option == 2:
                traffic = traffic * ((tvm.DataType(t_node_op.dtype).bits + 7) // 8)
        return traffic

    def calculate_set(self):
        func_psi_dict = TimeSeriesMaxStore()
        input_set_size = self.input_size
        output_set_size = self.output_size
        time_index = 0

        for node in self.nodes:
            for t_node_op in node.compute_ops:
                for t_source_node_op in t_node_op.input_tensors:
                    node_name = node.name
                    t_source_node_op_name = t_source_node_op.name
                    if t_source_node_op.name[:5] == "input":
                        index = int(t_source_node_op.name[5:])
                        if index < len(node.inputs):
                            src_node = node.inputs[index].src_node
                            if src_node in self.nodes:
                                node_name = src_node.name
                                t_source_node_op_name = "output0"

                    t_sub_size = self.calculate_input_tensor_size(node, t_source_node_op, self.shape_map, 2)
                    self.cache.op(node_name + "_" + t_source_node_op_name, t_sub_size)
                    t_gen_func_eta = self.cache.memory_transfer_cache.get(node_name + "_" + t_source_node_op_name, 0)
                    func_psi_dict.insert(node_name + "_" + t_source_node_op_name, time_index, t_gen_func_eta)

                for i in range(t_node_op.num_outputs):
                    node_name = node.name
                    t_node_op_output = t_node_op.output(i)
                    t_sub_size = self.calculate_output_tensor_size(node, t_node_op_output, self.shape_map, 2)
                    self.cache.op(node_name + "_" + t_node_op_output.name, t_sub_size, 1)
                time_index += 1
        self.cache.clear_all()

        H_R, H_BR, W_R, W_RB, W_R_cap_W_RB = 0, torch.inf, 0, torch.inf, torch.inf
        sum_output = 0
        sum_input = 0
        time_index = 0
        for node in self.nodes:
            t_H_R, t_H_BR, t_W_R, t_W_RB, t_W_R_cap_W_RB = 0, 0, 0, 0, 0
            for t_node_op in node.compute_ops:
                for t_source_node_op in t_node_op.input_tensors:
                    node_name = node.name
                    t_source_node_op_name = t_source_node_op.name
                    if t_source_node_op.name[:5] == "input":
                        index = int(t_source_node_op.name[5:])
                        if index < len(node.inputs):
                            src_node = node.inputs[index].src_node
                            if src_node in self.nodes:
                                node_name = src_node.name
                                t_source_node_op_name = "output0"

                    t_sub_size = self.calculate_input_tensor_size(node, t_source_node_op, self.shape_map, 2)
                    t_sub_size2 = self.calculate_input_tensor_size(node, t_source_node_op, self.shape_map, 1)
                    if t_sub_size != 0:
                        factor = t_sub_size2 / t_sub_size
                    else:
                        factor = 1

                    t_H_BR += (self.cache.file_sizes[node_name + "_" + t_source_node_op_name] -
                             self.cache.cache.get(node_name + "_" + t_source_node_op_name, 0)) * factor
                    t_H_R += self.cache.cache.get(node_name + "_" + t_source_node_op_name, 0) * factor
                    t_W_RB_1, name_list, size_list = self.cache.op(node_name + "_" + t_source_node_op_name, t_sub_size)
                    t_W_RB_2 = (self.cache.file_sizes[node_name + "_" + t_source_node_op_name] -
                             self.cache.cache.get(node_name + "_" + t_source_node_op_name, 0))
                    t_W_RB += (t_W_RB_1 + t_W_RB_2) * factor
                    t_W_R += self.cache.cache.get(node_name + "_" + t_source_node_op_name, 0) * factor
                    #t_W_R += self.cache.cache.get(node_name + "_" + t_source_node_op_name, 0) * factor
                    #t_W_R_cap_W_RB += min(func_psi_dict.query(node_name + "_" + t_source_node_op_name, time_index),t_W_RB_2)
                    for i in range(len(name_list)):
                        t_W_R_cap_W_RB += min(func_psi_dict.query(name_list[i], time_index), size_list[i]) * factor

                # output
                for i in range(t_node_op.num_outputs):
                    node_name = node.name
                    t_node_op_output = t_node_op.output(i)
                    t_sub_size = self.calculate_output_tensor_size(node, t_node_op_output, self.shape_map, 2)
                    t_sub_size2 = self.calculate_output_tensor_size(node, t_node_op_output, self.shape_map, 1)
                    if t_sub_size != 0:
                        factor = t_sub_size2 / t_sub_size
                    else:
                        factor = 1

                    t_W_RB_1, name_list, size_list = self.cache.op(node_name + "_" + t_node_op_output.name, t_sub_size, 1)
                    t_W_RB_2 = (self.cache.file_sizes[node_name + "_" + t_node_op_output.name] -
                             self.cache.cache.get(node_name + "_" + t_node_op_output.name, 0))
                    t_W_RB += (t_W_RB_1 + t_W_RB_2) * factor
                    t_W_R += self.cache.cache.get(node_name + "_" + t_node_op_output.name, 0) * factor
                    #t_W_R_cap_W_RB += min(func_psi_dict.query(node_name + "_" + t_node_op_output.name, time_index), t_W_RB_2)
                    for i in range(len(name_list)):
                        t_W_R_cap_W_RB += min(func_psi_dict.query(name_list[i], time_index), size_list[i]) * factor

                time_index += 1
                H_R = max(H_R, t_H_R)
                W_R = max(W_R, t_W_R)
                H_BR = min(H_BR, t_H_BR)
                W_RB = min(W_RB, t_W_RB)
                W_R_cap_W_RB = min(W_R_cap_W_RB, t_W_R_cap_W_RB)
                sum_input += t_H_R + t_H_BR
                sum_output += max(t_W_R + t_W_RB - t_W_R_cap_W_RB,0)

        self.cache.clear_all()
        X_value_candidate = max(max(input_set_size, sum_input), max(output_set_size, sum_output))
        return H_R, H_BR, W_R, W_RB, W_R_cap_W_RB, X_value_candidate

    def compute_IO_complexity(self):
        # Use X1,X2-Partition Theorem to compute I/O complexity
        R1, T1, R2, T2, T3, X_candi_value = self.calculate_set()
        X_candi_value = max(X_candi_value, self.capacity)

        if R2 - T1 - T3 > R1 - T2:
            formula_remain = -R2 + (T1 + T3)
        else:
            formula_remain = -R1 + T2

        _rho = max(1e-4, self.calculate_computational_density(formula_remain, X_candi_value))
        return _rho, self.sub_computation_size, X_candi_value, formula_remain


    def calculate_computational_density(self, formula_remain, X_candi_value):
        H_max = self.sub_computation_size
        self.X_values = X_candi_value
        return float(H_max/((X_candi_value + formula_remain + 1)))
