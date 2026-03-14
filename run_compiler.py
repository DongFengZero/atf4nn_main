import argparse
import time
import torch
import numpy as np
import main
from main import arch
from main.Engine import (Engine, MultiProcTunner, Tunner, load_model,
                           save_results)
from main.ATF4NN.DQN.Game import Game
from main.Engine.base_tunner import _extract_subgraph, subgraph_hash, eliminate_memcpy

if __name__ == "__main__":
    op = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--arch', type=str, default="g4090")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--nofusion', action="store_true")
    parser.add_argument('--max_fusion_node', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=500)

    args = parser.parse_args()
    main.set_log_level(args.verbose)
    assert args.input_file.endswith(".json")
    start_time = time.time()
    ordered_nodes = load_model(args.input_file)

    # tunner = Tunner(arch=arch.__getattribute__(args.arch)(), device="cuda:{}".format(args.device), topk=args.topk, check=args.check)
    tunner = MultiProcTunner(input_file_path=args.input_file,
                             arch=arch.__getattribute__(args.arch)(), device="cuda:{}".format(args.device),
                             topk=args.topk, check=args.check)
    engine = Engine(tunner)
    # if args.nofusion:
    # fusion_groups = engine.run(ordered_nodes)
    # else:
    #     fusion_groups = engine.run(ordered_nodes)
    #     t_fusion_groups, t_fusion_groups_2, length, index_to_node, node_to_index = engine.run(ordered_nodes)
    #     # for i in range(len(t_fusion_groups_2)):
    #     #     print(t_fusion_groups_2[i].nodes)
    length = 0
    node_to_index = {}
    for node in ordered_nodes:
        if node.is_output() or node.is_placeholder():
            continue
        node_to_index[node] = length
        length += 1

    RLop = Game(length, args.model_name, max_fusing_node=min(args.max_fusion_node,len(ordered_nodes)), epoch=args.epoch)
    fusion_groups = RLop.train(ordered_nodes, node_to_index, engine)

    ##添加强化学习算法

    gain = sum([fg.gain for fg in fusion_groups])
    print("Fusion gain: {}ms".format(gain))
    if args.output_file != "":
        save_results(fusion_groups, args.output_file)
    print("Total run time: ", time.time() - start_time)
