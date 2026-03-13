import numpy as np
import torch

def are_tuples_equal(tuple1, tuple2):
    # 处理None情况
    if tuple1 is None and tuple2 is None:
        return True
    if tuple1 is None or tuple2 is None:
        return False
    
    if len(tuple1) != len(tuple2):
        return False
    
    for a, b in zip(tuple1, tuple2):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True

def find_tuple_index(data, tuple_list, valid_count):
    """只在有效数据范围内搜索"""
    for idx in range(valid_count):
        if are_tuples_equal(data, tuple_list[idx]):
            return idx
    return -1

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
    
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def add(self, p, data):
        # 只在有效数据中查找
        idx = find_tuple_index(data, self.data, self.n_entries)
        if idx != -1:
            tree_idx = idx + self.capacity - 1
            self.update(tree_idx, p)
            # 数据已存在，只更新权重
            return
        
        # 添加新数据
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 使用 < 而不是 <=，避免边界问题
                if v < self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total(self):
        return self.tree[0]
    
    def len(self):
        return self.n_entries

class MaxSumTree(SumTree):
    def __init__(self, capacity: int):
        super().__init__(capacity)
    
    def add(self, p, data):
        idx = find_tuple_index(data, self.data, self.n_entries)
        if idx != -1:
            tree_idx = idx + self.capacity - 1
            new_p = max(p, self.tree[tree_idx])
            self.update(tree_idx, new_p)
            # 如果需要，也更新data
            # self.data[idx] = data
            return
        super().add(p, data)