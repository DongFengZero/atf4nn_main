from typing import List
import math
import numpy as np


def get_all_factors(n: int) -> List[int]:
    n0 = int(np.ceil(np.sqrt(n)))
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]

def factorize(n: int) -> List[int]:
    i = 2
    result = []
    while n > 1:
        if n % i == 0:
            n //= i
            result.append(i)
        else:
            i += 1
    return result

def coalesced_tensor_shape2(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    bytes = int(np.prod(subtensor))
    if bytes == 0: return 0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)

def coalesced_subtensor_shape(subtensor: List[int], tensor: List[int]) -> int:
    # 计算subtensor的大小
    bytes = int(np.prod(subtensor))

    # 如果bytes为0，直接返回0
    if bytes == 0:
        return 0

    result = 0
    # 计算加权累积
    for i in range(len(subtensor)):
        if i!=len(subtensor)-1:
            result += subtensor[i] * np.prod(tensor[i+1:])
        else:
            result += subtensor[i]

    return result

def coalesced_tensor_shape3(subtensor: List[int], tensor: List[int]) -> float:
    return float(1.0/coalesced_factor(subtensor, tensor))

def coalesced_tensor_shape5(subtensor: List[int], tensor: List[int]) -> int:
    # 计算subtensor的大小
    bytes = int(np.prod(subtensor))

    # 如果bytes为0，直接返回0
    if bytes == 0:
        return 0

    result_max = 0
    result_min = float('inf')
    # 计算加权累积
    for i in reversed(range(len(subtensor))):
        result_max = max(result_max, tensor[i]/subtensor[i])
        result_min = min(result_min, tensor[i]/subtensor[i])

    return result_max - result_min

def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    if int(subtensor[-1]) != int(tensor[-1]) or len(subtensor) == 1:
        return int(subtensor[-1])
    else:
        return int(subtensor[-1]) * coalesced_factor(subtensor[:-1], tensor[:-1])

def coalesced_tensor_shape4(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    bytes = int(np.prod(subtensor))
    # return bytes
    if bytes == 0: return 0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)

def coalesced_tensor_shape(subtensor: List[int], tensor=None, transaction_size=None) -> int:
    bytes = int(np.prod(subtensor))
    # return bytes
    if bytes == 0: return 0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)

def coalesced_tensor_shape41(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    bytes = int(np.prod(subtensor))
    # return bytes
    if bytes == 0: return 0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)



