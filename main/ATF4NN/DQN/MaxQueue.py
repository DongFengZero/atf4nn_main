import torch
import heapq
import itertools

class MaxTupleArray:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.tuple_array = []  # 存储 (objective_value, tuple)
        self.existing_tuples = set()  # 用于去重的集合
        self.counter = itertools.count()

    def insert(self, new_tuple, objective_value):
        # 将 new_tuple 转换为 tuple,以便它是可哈希的
        new_tuple_as_tuple = tuple(new_tuple)
        
        # 先检查是否重复
        if new_tuple_as_tuple in self.existing_tuples:
            return
        
        count = next(self.counter)
        # 容量未满,直接插入
        if len(self.tuple_array) < self.capacity:
            heapq.heappush(self.tuple_array, (objective_value, count, new_tuple_as_tuple))
            self.existing_tuples.add(new_tuple_as_tuple)
        # 容量已满,检查是否需要替换最小值
        elif objective_value > self.tuple_array[0][0]:  # 堆顶是最小值
            old_value, _, old_tuple = heapq.heappop(self.tuple_array)
            self.existing_tuples.remove(old_tuple)
            heapq.heappush(self.tuple_array, (objective_value, count, new_tuple_as_tuple))
            self.existing_tuples.add(new_tuple_as_tuple)
    
    def get_maximums(self):
        # 返回按objective_value降序排列的tuples
        return [tup[2] for tup in sorted(self.tuple_array, key=lambda x: x[0], reverse=True)]
    
    def get_maximums_reward(self):
        # 返回按降序排列的objective_values
        return [tup[0] for tup in sorted(self.tuple_array, key=lambda x: x[0], reverse=True)]
