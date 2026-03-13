import onnx
import torch
import numpy as np
from onnx import helper, TensorProto
from transformers import AutoModelForCausalLM, AutoConfig
from torch.onnx import register_custom_op_symbolic

def sdpa_symbolic(g, query, key, value, attn_mask, dropout, is_causal, scale):
    scores = g.op('MatMul', query, g.op('Transpose', key, perm_i=[0,1,3,2]))
    weights = g.op('Softmax', scores, axis_i=-1)
    return g.op('MatMul', weights, value)
register_custom_op_symbolic('aten::scaled_dot_product_attention', sdpa_symbolic, 11)

config = AutoConfig.from_pretrained('facebook/opt-125m', trust_remote_code=True)
config.use_cache = False
config.num_hidden_layers = 4
config.hidden_size = 512
config.intermediate_size = 1024
config.num_attention_heads = 8
config.num_key_value_heads = 8
config._attn_implementation = 'eager'
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()

input_ids = torch.randint(0, config.vocab_size, (96, 128))
attention_mask = torch.ones(96, 128, dtype=torch.long)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    './temp_model/qwen2/model.onnx',
    opset_version=11,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={}
)

# 移除 ScatterND 节点，替换为直接透传 data 输入
model = onnx.load('./temp_model/qwen2/model.onnx')
nodes_to_remove = []
replace_map = {}
for node in model.graph.node:
    if node.op_type == 'ScatterND':
        # ScatterND(data, indices, updates) -> 直接用 data
        replace_map[node.output[0]] = node.input[0]
        nodes_to_remove.append(node)
for node in nodes_to_remove:
    model.graph.node.remove(node)
# 替换所有引用了 ScatterND 输出的节点输入
for node in model.graph.node:
    for i, inp in enumerate(node.input):
        if inp in replace_map:
            node.input[i] = replace_map[inp]
onnx.save(model, './temp_model/qwen2/model.onnx')

feed_dict = {'input_ids': input_ids.numpy(), 'attention_mask': attention_mask.numpy()}
np.savez('./temp_model/qwen2/inputs.npz', **feed_dict)
print('ScatterND removed, export done')
