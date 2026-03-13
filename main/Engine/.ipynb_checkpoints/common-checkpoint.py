import json
import numpy as np
from typing import List

from ..graph import Node
from ..utils import CompileResult


def get_id_by_name(name):
    return int(name.split("_")[-1])

class FusionGroup():
    def __init__(self, node_list: List[Node], group_id: int, cpresult: "CompileResult", gain: float) -> None:
        self.nodes = node_list
        self.group_id = group_id
        self.cpresult = cpresult
        self.gain = gain

def dump(fusion_groups: List[FusionGroup]):
    obj = []
    result_reuse_map = {}
    for group in fusion_groups:
        group_desc = {}
        node_names = [node.name for node in group.nodes]
        group_desc["nodes"] = [get_id_by_name(name) for name in node_names]
        group_desc["node_names"] = node_names
        group_desc["group_id"] = group.group_id
        if group.cpresult is not None:
            cpresult = group.cpresult
            group_desc["input_desc"] = [[get_id_by_name(name), id] for name, id in cpresult.input_desc]
            group_desc["output_desc"] = [[get_id_by_name(name), id] for name, id in cpresult.output_desc]

            if cpresult.origin in result_reuse_map:
                cpresult = result_reuse_map[cpresult.origin]
            else:
                result_reuse_map[cpresult.origin] = cpresult
            group_desc["code"] = cpresult.code
            group_desc["block_size"] = cpresult.block_size
            group_desc["grid_size"] = cpresult.grid_size
            group_desc["latency"] = cpresult.latency
            group_desc["name"] = cpresult.name
            group_desc["gain"] = group.gain
        obj.append(group_desc)
    return obj

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

def save_results(fusion_groups: List[FusionGroup], fname: str):
    obj = dump(fusion_groups)
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)
    return None

__all__ = ['save_results', 'FusionGroup']
