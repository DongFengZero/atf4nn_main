import numpy as np
import random
import itertools
import copy
from ..policy.common import coalesced_tensor_shape3, coalesced_subtensor_shape
class GeneticAlgorithmOptimizer:
    def __init__(self, objective_function, init_tile, population_size=1000, init_population_size=2500, dimension=5,
                 lower_bound=None, upper_bound=None, generations=1000, mutation_rate=0.75, rstep_map=None,
                 range=64, factor=8, max_times=1000, node=None, thd=None, topK=10, flag=False):
        """
        初始化遗传算法优化器
        :param objective_function: 目标函数，输入参数组合，返回td结构，td结构包含traffic作为评价指标
        :param population_size: 种群大小
        :param dimension: 参数维度
        :param lower_bound: 参数下界（必须是长度为dimension的数组）
        :param upper_bound: 参数上界（必须是长度为dimension的数组）
        :param generations: 最大迭代次数
        :param mutation_rate: 变异率
        """
        self.objective_function = objective_function
        self.population_size = population_size
        self.init_population_size = init_population_size
        self.dimension = dimension
        self.init_tile = init_tile
        self.cache = {}  # 用于缓存计算结果
        self.range = range
        self.topK = topK
        self.factor = factor
        self.max_times = max_times
        self.node = node
        self.thd = thd

        # 如果未提供上下界，则默认为 [1, 1, ..., 1]
        self.lower_bound = np.array(lower_bound if lower_bound is not None else [1] * dimension)
        self.upper_bound = np.array(upper_bound if upper_bound is not None else [1] * dimension)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rstep_map = rstep_map
        self.init_population = []
        self.flag = flag
        self.upper_bound_factors = [self.calculate_factors(ub) for ub in self.upper_bound]


    def calculate_factors(self, number):
        # 计算数字的所有因数
        factors = [i for i in range(1, number + 1) if number % i == 0]
        return factors

    def cached_objective_function(self, individual):
        # 将个体转换为元组以便在字典中使用
        individual_tuple = tuple(individual)

        # 检查缓存中是否已有此结果
        if individual_tuple in self.cache:
            return self.cache[individual_tuple]

        # 如果缓存中没有，则计算结果并检查合法性
        flag = True
        td = self.objective_function(np.array(individual), self.rstep_map, flag)
        if td.valid:  # 仅缓存合法的 td 结果
            self.cache[individual_tuple] = td
        else:
            td = None  # 如果不合法，则返回 None
            self.cache[individual_tuple] = td
        return td

    def enumerate_all_points(self):
        # 枚举所有可能的整点组合
        select_factor = []
        for j in range(self.dimension):
            # 生成范围内的可能值
            factor_multiples = [k * self.factor for k in range((self.lower_bound[j] + self.factor - 1) // self.factor,
                                                               (min(self.upper_bound[j], 64) // self.factor) + 1)]
            powers_of_2 = [2 ** k for k in range(int(np.log2(self.upper_bound[j])) + 1) if
                           self.lower_bound[j] <= 2 ** k <= self.upper_bound[j]]
            upper_factors = [f for f in self.upper_bound_factors[j] if
                             self.lower_bound[j] <= f <= self.upper_bound[j]]
            if self.flag:
                possible_values = sorted(set(upper_factors))
            else:
                possible_values = sorted(set(factor_multiples + powers_of_2 + upper_factors))
            select_factor.append(possible_values)

        all_points = list(itertools.product(*select_factor))
        all_points.sort(key=lambda x: coalesced_tensor_shape3(list(x), self.upper_bound))
        unique_td_population = set()  # 使用集合来去重
        results = []

        for point in all_points:
            td = self.cached_objective_function(np.array(point))  # 使用缓存机制
            if td and td.valid and td not in unique_td_population:  # 确保td合法且不重复
                unique_td_population.add(td)
                results.append((td, point))

        # 根据traffic评分进行排序
        results.sort(key=lambda x: (x[0].traffic, x[0].factor1, x[0].factor))
        best_td_population = [result[0] for result in results[:self.topK]]
        best_fitness_values = [result[0].traffic for result in results[:self.topK]]
        best_individuals = [result[1] for result in results[:self.topK]]

        return best_td_population, best_fitness_values, best_individuals

    def initialize_population(self):
        population = set()
        index = 0
        select_factor = []
        for j in range(self.dimension):
            # 生成范围内的可能值
            factor_multiples = [k * self.factor for k in range((self.lower_bound[j] + self.factor - 1) // self.factor,
                                                               (min(self.upper_bound[j], 64) // self.factor) + 1)]
            powers_of_2 = [2 ** k for k in range(int(np.log2(self.upper_bound[j])) + 1) if
                           self.lower_bound[j] <= 2 ** k <= self.upper_bound[j]]
            upper_factors = [f for f in self.upper_bound_factors[j] if
                             self.lower_bound[j] <= f <= self.upper_bound[j]]
            if self.flag:
                possible_values = sorted(set(upper_factors))
            else:
                possible_values = sorted(set(factor_multiples + powers_of_2 + upper_factors))
            select_factor.append(possible_values)

        t_population = list(itertools.product(*select_factor))
        t_population.sort(key=lambda x: coalesced_tensor_shape3(list(x), self.upper_bound))

        # 获取前 n 个组合
        top_n_combinations = t_population[:self.init_population_size]
        population.update(top_n_combinations)

        while len(population) < self.init_population_size:
            individual = []
            for j in range(self.dimension):
                # 生成范围内的可能值
                factor_multiples = [k * self.factor for k in range((self.lower_bound[j] + self.factor - 1) // self.factor,
                                                                   (min(self.upper_bound[j], 64) // self.factor) + 1)]
                powers_of_2 = [2 ** k for k in range(int(np.log2(self.upper_bound[j])) + 1) if
                               self.lower_bound[j] <= 2 ** k <= self.upper_bound[j]]
                upper_factors = [f for f in self.upper_bound_factors[j] if
                                 self.lower_bound[j] <= f <= self.upper_bound[j]]

                # 合并并去重
                if self.flag:
                    possible_values = sorted(set(upper_factors))
                else:
                    possible_values = sorted(set(factor_multiples + powers_of_2 + upper_factors))

                # 确保值在上下界范围内
                if possible_values:
                    selected_value = np.random.choice(possible_values)
                else:
                    selected_value = self.lower_bound[j]  # 如果没有可能值，则使用下界

                # 确保最终选值在上下界内
                selected_value = max(self.lower_bound[j], min(selected_value, self.upper_bound[j]))
                individual.append(selected_value)

            population.add(tuple(individual))
            index += 1
            if index == 2 * self.init_population_size:
                break

        return np.array(list(population), dtype=int)

    import numpy as np

    def evaluate_fitness(self, population):
        valid_population = []
        td_population = []
        fitness_scores = []
        #traffic1_scores = []

        for individual in population:
            td = self.cached_objective_function(individual)
            if td is not None and td.valid:  # 只保留合法的td
                valid_population.append(individual)
                td_population.append(td)
                fitness_scores.append(td.traffic)

        # 将 fitness_scores 转换为 numpy 数组以便排序
        fitness_scores = np.array(fitness_scores)

        # 获取排序的索引，按照从小到大的顺序
        sorted_indices = np.argsort(fitness_scores)

        # 根据排序后的索引重新排列 valid_population、td_population 和 fitness_scores
        sorted_valid_population = np.array(valid_population)[sorted_indices]
        sorted_td_population = [td_population[i] for i in sorted_indices]
        sorted_fitness_scores = fitness_scores[sorted_indices]

        return sorted_valid_population, sorted_td_population, sorted_fitness_scores

    def select_parents(self, population, fitness, num_parents):
        max_fitness = fitness.max()
        fitness2 = (max_fitness - fitness)
        prob = fitness2 / fitness2.sum()
        random_parents = population[np.random.choice(len(population), size=int(num_parents), p=prob)]
        return random_parents

    def crossover(self, parents, offspring_size):
        offspring = []
        for _ in range(offspring_size):
            parent1, parent2 = random.sample(list(parents), 2)
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)
        return np.array(offspring, dtype=int)  # 确保子代为整数

    def mutate(self, offspring):
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                for j in range(len(offspring[i])):
                    # 生成附近值的范围
                    lower_bound = max(self.lower_bound[j], offspring[i][j] - self.range)
                    upper_bound = min(self.upper_bound[j], offspring[i][j] + self.range)

                    # 生成可能值
                    factor_multiples = [k * self.factor for k in
                                        range((lower_bound + self.factor - 1) // self.factor,
                                              (min(upper_bound, 64) // self.factor) + 1)]
                    powers_of_2 = [2 ** k for k in range(int(np.log2(upper_bound)) + 1) if
                                   lower_bound <= 2 ** k <= upper_bound]
                    upper_factors = [f for f in self.upper_bound_factors[j] if lower_bound <= f <= upper_bound]

                    # 合并并去重
                    if self.flag:
                        possible_values = sorted(set(upper_factors))
                    else:
                        possible_values = sorted(set(factor_multiples + powers_of_2 + upper_factors))
                        
                    if offspring[i][j] in possible_values:
                        possible_values.remove(offspring[i][j])

                    # 随机选择一个值并限制在上下界内
                    if possible_values:
                        new_value = random.choice(possible_values)
                        new_value = max(min(new_value, self.upper_bound[j]), self.lower_bound[j])
                        offspring[i][j] = new_value

        return np.array(offspring, dtype=int)

    def optimize(self):
        total_points = np.prod(np.ceil((self.upper_bound - self.lower_bound + 1)))
        if total_points < 2000:
            return self.enumerate_all_points()

        population = self.initialize_population()
        f_population = []
        best_population = []
        blank_epoch = 0
        last_f_population = 0

        for generation in range(self.generations):
            new_population = len(f_population) - last_f_population
            print("generation：",generation,"population:",len(f_population)," upper_bound:",self.upper_bound," lower_bound:",self.lower_bound)
            population, td_population, fitness = self.evaluate_fitness(population)
            if len(population) == 0:
                break
            print("best_td:", td_population[0].output_tile,
                  " best_num_wave:", td_population[0].num_wave,
                  " best_grid_size:",td_population[0].grid_size,
                  " best_traffic:", td_population[0].traffic,
                  " best_old_traffic:", td_population[0].traffic1)
            if new_population == 0:
                blank_epoch += 1
            else:
                blank_epoch = 0

            if blank_epoch == 5:
                break
            # else:
            #     print("find new population:",new_population)

            last_f_population = len(f_population)
            if len(f_population) == 0:
                f_population = copy.deepcopy(population)
            else:
                f_population = np.vstack((f_population, copy.deepcopy(population)))
                f_population = np.unique(f_population, axis=0)

            if len(best_population) == 0:
                best_population = copy.deepcopy(population[:self.topK])
            else:
                best_population = np.vstack((best_population, copy.deepcopy(population[:self.topK])))
                best_population = np.unique(best_population, axis=0)

            if len(population) <= self.topK:
                break

            parents = self.select_parents(population, fitness, num_parents=int(np.ceil(self.population_size/2)))
            offspring_size = self.population_size - len(parents)
            offspring = self.crossover(parents, offspring_size)
            offspring = self.mutate(offspring)
            population = np.vstack((parents, offspring))  # 都非空时正常拼接
            population = np.unique(population, axis=0)


        population, td_population, final_fitness = self.evaluate_fitness(best_population)
        unique_results = {}

        for idx in range(len(td_population)):
            individual = tuple(population[idx])
            traffic_score = final_fitness[idx]
            if individual not in unique_results or unique_results[individual].traffic > traffic_score:
                unique_results[individual] = td_population[idx]

        visited_tiles = filter(lambda x: x[1].valid, unique_results.items())
        sorted_results = sorted(visited_tiles, key=lambda x:(x[1].traffic, x[1].factor1, x[1].factor))
        best_td_population = [result[1] for result in sorted_results[:self.topK]]
        best_fitness_values = [result[1].traffic for result in sorted_results[:self.topK]]
        best_individuals = [result[0] for result in sorted_results[:self.topK]]

        return best_td_population, best_fitness_values, best_individuals

class GeneticAlgorithmOptimizer_thread(GeneticAlgorithmOptimizer):
    def cached_objective_function(self, individual):
        # 将个体转换为元组以便在字典中使用
        individual_tuple = tuple(individual)

        # 检查缓存中是否已有此结果
        if individual_tuple in self.cache:
            return self.cache[individual_tuple]

        # 如果缓存中没有，则计算结果并检查合法性
        flag = True
        td = self.objective_function(self.node, np.array(individual), self.thd)
        self.cache[individual_tuple] = td
        if td.valid:  # 仅缓存合法的 td 结果
            self.cache[individual_tuple] = td
        else:
            td = None  # 如果不合法，则返回 None
            self.cache[individual_tuple] = td
        return td

    def optimize(self):
        total_points = np.prod(np.ceil((self.upper_bound - self.lower_bound + 1)))
        if total_points < 2000:
            return self.enumerate_all_points()

        population = self.initialize_population()
        f_population = []
        best_population = []
        blank_epoch = 0
        last_f_population = 0

        for generation in range(self.generations):
            new_population = len(f_population) - last_f_population
            # print("generation：",generation,"population:",len(f_population)," upper_bound:",self.upper_bound," lower_bound:",self.lower_bound)
            population, td_population, fitness = self.evaluate_fitness(population)
            if len(population) == 0:
                break
            # print("best_td:", td_population[0].thread,
            #       " best_traffic:", td_population[0].traffic,
            #       " best_factor1:", td_population[0].factor1,
            #       " best_factor:", td_population[0].factor
            #      )
            if new_population == 0:
                blank_epoch += 1
            else:
                blank_epoch = 0

            if blank_epoch == 5:
                break
            # else:
            #     print("find new population:",new_population)

            last_f_population = len(f_population)
            if len(f_population) == 0:
                f_population = copy.deepcopy(population)
            else:
                f_population = np.vstack((f_population, copy.deepcopy(population)))
                f_population = np.unique(f_population, axis=0)

            if len(best_population) == 0:
                best_population = copy.deepcopy(population[:self.topK])
            else:
                best_population = np.vstack((best_population, copy.deepcopy(population[:self.topK])))
                best_population = np.unique(best_population, axis=0)

            if len(population) <= self.topK:
                break

            parents = self.select_parents(population, fitness, num_parents=int(np.ceil(self.population_size/2)))
            offspring_size = self.population_size - len(parents)
            offspring = self.crossover(parents, offspring_size)
            offspring = self.mutate(offspring)
            population = np.vstack((parents, offspring))  # 都非空时正常拼接
            population = np.unique(population, axis=0)


        population, td_population, final_fitness = self.evaluate_fitness(best_population)
        unique_results = {}

        for idx in range(len(td_population)):
            individual = tuple(population[idx])
            traffic_score = final_fitness[idx]
            if individual not in unique_results or unique_results[individual].traffic > traffic_score:
                unique_results[individual] = td_population[idx]

        visited_tiles = filter(lambda x: x[1].valid, unique_results.items())
        sorted_results = sorted(visited_tiles, key=lambda x:(x[1].traffic, x[1].factor1, x[1].factor))
        best_td_population = [result[1] for result in sorted_results[:self.topK]]
        best_fitness_values = [result[1].traffic for result in sorted_results[:self.topK]]
        best_individuals = [result[0] for result in sorted_results[:self.topK]]

        return best_td_population, best_fitness_values, best_individuals

    
    
