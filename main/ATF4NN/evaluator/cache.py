from collections import OrderedDict

class FileCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = OrderedDict()  # LRU Cache
        self.memory = {}  # 无限大小的慢速内存
        #self.cache_transfer_memory = {}
        self.memory_transfer_cache = {}
        self.file_sizes = {}  # 文件完整大小
        self.modify_flag = {}
        self.write_file_sizes = {}

    def find_file_size(self, file_name):
        if file_name in self.file_sizes:
            return True
        return False

    def set_file_size(self, file_name, size):
        if file_name not in self.file_sizes:
            self.file_sizes[file_name] = size
            self.modify_flag[file_name] = 0
            self.write_file_sizes[file_name] = 0

    def get_file_size(self, file_name):
        return self.file_sizes.get(file_name, None)

    def update_value(self, map, key, new_value):
        map[key] = new_value


    def op(self, file_name, size, flag=0):
        if file_name not in self.file_sizes:
            self.file_sizes[file_name] = size

        if flag == 1:
            self.modify_flag[file_name] = 1
            self.write_file_sizes[file_name] = size
        else:
            self.modify_flag[file_name] = 0
            self.write_file_sizes[file_name] = 0

        total_size = self.file_sizes[file_name]
        current_cache_size = self.cache.get(file_name, 0)

        # 计算实际存储大小
        size_to_store = total_size - current_cache_size
        cache_transfer_memory = 0
        output_name_list = []
        output_size = []
        if size_to_store > 0:
            self.cache[file_name] = current_cache_size + size_to_store
            self.memory[file_name] = total_size - size_to_store
            self.cache.move_to_end(file_name)  # 标记为最近使用
            self.update_value(self.memory_transfer_cache, file_name, size_to_store)

            # 如果缓存总大小超过限制，进行 LRU 替换
            #cache_transfer_memory = max(sum(self.cache.values())-self.cache_size,0)
            while sum(self.cache.values()) > self.cache_size:
                oldest_file, old_size = next(iter(self.cache.items()))
                temp_size = sum(self.cache.values()) - self.cache_size
                if oldest_file == file_name:
                    self.cache[oldest_file] -= temp_size
                    self.memory[oldest_file] += temp_size
                    output_name_list.append(oldest_file)
                    if flag == 1:
                        out_size = min(temp_size, self.write_file_sizes[oldest_file])
                        output_size.append(min(temp_size, self.write_file_sizes[oldest_file]))
                        self.write_file_sizes[oldest_file] -= out_size
                        cache_transfer_memory += out_size
                    else:
                        output_size.append(0)
                    break
                elif old_size <= temp_size:
                    del self.cache[oldest_file]
                    self.memory[oldest_file] += old_size
                    output_name_list.append(oldest_file)
                    if self.modify_flag[oldest_file] == 1:
                        out_size = min(temp_size, self.write_file_sizes[oldest_file])
                        output_size.append(min(temp_size, self.write_file_sizes[oldest_file]))
                        self.write_file_sizes[oldest_file] -= out_size
                        cache_transfer_memory += out_size
                    else:
                        output_size.append(0)
                    #output_size.append(old_size)
                else:
                    self.cache[oldest_file] -= temp_size
                    self.memory[oldest_file] += temp_size
                    output_name_list.append(oldest_file)
                    if self.modify_flag[oldest_file] == 1:
                        out_size = min(temp_size, self.write_file_sizes[oldest_file])
                        output_size.append(min(temp_size, self.write_file_sizes[oldest_file]))
                        self.write_file_sizes[oldest_file] -= out_size
                        cache_transfer_memory += out_size
                    else:
                        output_size.append(0)
                    #output_size.append(temp_size)
                    break

        return cache_transfer_memory, output_name_list, output_size


    def query_cache_size(self, file_name):
        return self.cache.get(file_name, 0)

    def query_memory_size(self, file_name):
        return self.memory.get(file_name, 0)

    def clear_all(self):
        self.cache.clear()
        self.memory.clear()
        self.memory_transfer_cache.clear()
        self.write_file_sizes.clear()
        self.modify_flag.clear()