import numpy as np
import torch

def edgenext(batch_size, model_name='edgenext_xx_small'):
    import timm
    model = timm.create_model(model_name, pretrained=False)
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input,)

def efficientnet(batch_size, model_name='efficientnetv2_l'):
    import timm
    model = timm.create_model(model_name, pretrained=False)
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input,)

def vgg11bn(batch_size, model_name='vgg11_bn'):
    import timm
    model = timm.create_model(model_name, pretrained=False)
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input,)

def StarNet(batch_size, model_name='starnet_s1'):
    import timm
    model = timm.create_model(model_name, pretrained=False)
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input,)

def shufflenet(batch_size):
    from torchvision.models import shufflenet_v2_x1_0 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def swin_transformer(batch_size):
    import timm
    model = timm.create_model('swinv2_base_window8_256', pretrained=False)
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input,)

def bert(batch_size):
    from .bert_config import BertConfig
    from .pytorch_bert import BertModel

    # from transformers import BertConfig, BertModel
    config = BertConfig(vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=128,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                batch_size=batch_size)
    model = BertModel(config)
    input_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    token_type_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    attention_mask = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    masked_lm_labels = None # torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    next_sentence_label = None # torch.LongTensor(np.ones([config.batch_size]))
    inputs = (input_ids, attention_mask, token_type_ids)
    # inputs = (input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
    return model, inputs

def NAFNet(batch_size):
    from .nafnet import NAFNet
    model = NAFNet(3, 16, 1, [1, 1, 1], [1, 1, 1])
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def mobilevit(batch_size):
    #import timm
    #imodel = timm.create_model('mobilevit_s', pretrained=False)
    from .mobilevit import mobilevit_xxs
    model = mobilevit_xxs()
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def NeRF(batch_size):
    from .mlp import MLP

    # 定义MLP
    model = MLP(
        batch_size=batch_size * 256 * 256,
        in_dim=64,
        out_dim=3,
        hidden_dim=64,
        n_layers=7
    )
    
    # 随机输入模拟
    input = torch.randn(batch_size * 256 * 256, 64)

    return model, (input,)

from transformers import AutoModelForCausalLM, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── 主动 import 所有可能含 ScatterND 的 Cache 类，然后立即替换 update ──────
def _passthrough_cache_update(self, key_states, value_states,
                              layer_idx, cache_kwargs=None):
    """直通版 update：不做任何 scatter 写入，直接返回当前 kv。"""
    return key_states, value_states

def _patch_all_cache_classes():
    # 显式 import，确保类已载入内存
    _cache_imports = [
        ("transformers",                    "SlidingWindowCache"),
        ("transformers",                    "StaticCache"),
        ("transformers",                    "DynamicCache"),
        ("transformers.cache_utils",        "SlidingWindowCache"),
        ("transformers.cache_utils",        "StaticCache"),
        ("transformers.cache_utils",        "DynamicCache"),
        ("transformers.models.mistral.modeling_mistral", "SlidingWindowCache"),
    ]
    patched = set()
    for mod_path, cls_name in _cache_imports:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "update") and cls not in patched:
                cls.update = _passthrough_cache_update
                patched.add(cls)
                print(f"[patch_cache] {mod_path}.{cls_name}.update → passthrough")
        except Exception:
            pass

    # 兜底：扫描 sys.modules 替换漏网之鱼
    import sys, inspect
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or "transformers" not in mod_name:
            continue
        for attr_name in list(vars(mod).keys()):
            try:
                cls = getattr(mod, attr_name)
            except Exception:
                continue
            if (not isinstance(cls, type) or "Cache" not in attr_name
                    or not hasattr(cls, "update") or cls in patched):
                continue
            try:
                src = inspect.getsource(cls.update)
                if any(op in src for op in ("index_copy_", "index_put_",
                                             "scatter_", "cumsum", "clamp")):
                    cls.update = _passthrough_cache_update
                    patched.add(cls)
                    print(f"[patch_cache] {mod_name}.{attr_name}.update → passthrough")
            except Exception:
                pass

_patch_all_cache_classes()
# =============================================================================
# ONNX opset-11 兼容 patch
# =============================================================================

class RMSNormCompat(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(
            weight.clone() if weight is not None else torch.ones(normalized_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def patch_rmsnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if "RMSNorm" in type(child).__name__:
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            setattr(module, name, RMSNormCompat(
                normalized_shape=child.weight.shape[0],
                eps=eps,
                weight=child.weight.data,
            ))
        else:
            patch_rmsnorm(child)


class StaticRoPECompat(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.head_dim = dim

    def forward(self, x: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        dtype, device = x.dtype, self.inv_freq.device
        if position_ids is not None:
            freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)
        t = torch.arange(seq_len, device=device).float()
        freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def patch_rope(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        cls_name = type(child).__name__
        if "RotaryEmbedding" in cls_name or "RopeEmbedding" in cls_name:
            dim = child.inv_freq.shape[0] * 2 if hasattr(child, "inv_freq") else \
                  getattr(child, "head_dim", 64)
            base = getattr(child, "base", 10000)
            setattr(module, name, StaticRoPECompat(dim=dim, base=base))
        else:
            patch_rope(child)


def patch_apply_rotary(model: nn.Module) -> None:
    import sys, inspect

    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def apply_rotary_pos_emb_compat(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if cos.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        seq = q.shape[-2]
        cos, sin = cos[..., :seq, :], sin[..., :seq, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    target_pkg = inspect.getmodule(type(model)).__name__.rsplit(".", 1)[0]
    for mod_name, mod in list(sys.modules.items()):
        if mod and mod_name.startswith(target_pkg) and hasattr(mod, "apply_rotary_pos_emb"):
            mod.apply_rotary_pos_emb = apply_rotary_pos_emb_compat
    for mod_name, mod in list(sys.modules.items()):
        if mod and "transformers" in mod_name and "modeling_" in mod_name \
                and hasattr(mod, "apply_rotary_pos_emb"):
            mod.apply_rotary_pos_emb = apply_rotary_pos_emb_compat


def patch_attention(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if "Attention" in type(child).__name__:
            for attr in ("sliding_window", "is_sliding", "attn_logit_softcapping"):
                if hasattr(child, attr):
                    setattr(child, attr, None)
        patch_attention(child)


def patch_causal_mask() -> None:
    def _triu_compat(tensor, diagonal=0):
        rows, cols = tensor.shape[-2], tensor.shape[-1]
        r = torch.arange(rows, device=tensor.device).unsqueeze(1)
        c = torch.arange(cols, device=tensor.device).unsqueeze(0)
        return tensor * ((c - r) >= diagonal).to(tensor.dtype)
    torch.triu = _triu_compat


def patch_causal_mask_generation(module: nn.Module) -> None:
    _noop = lambda *a, **kw: None
    for cls in type(module).__mro__:
        for method in ("_update_causal_mask",
                       "_prepare_decoder_attention_mask",
                       "_prepare_4d_causal_attention_mask"):
            if method in cls.__dict__:
                setattr(module, method, _noop)


# =============================================================================
# ★ ScatterND 根治：替换 SlidingWindowCache.update
#
# 调用链（已通过诊断确认）：
#
#   MistralModel.forward
#     → past_key_values = SlidingWindowCache(...)   ← Cache 在上层构造
#     → decoder_layer(past_key_value=past_key_values, ...)
#         → MistralAttention.forward(past_key_value=cache_obj, ...)
#             → if past_key_value is not None:           ← 非 None，进入
#                 past_key_value.update(...)             ← 触发 index_copy_
#                     → k_out.index_copy_(2, ...)        ← → ScatterND
#
# 为什么 patch Attention.forward 的 kwargs 方式无效：
#   past_key_value 以位置参数传入（args[3]），不在 kwargs 里，拦不到。
#
# 正确方案：直接替换 SlidingWindowCache.update 为直通版本。
#   直通语义：不做任何 scatter 写入，直接返回当前 key/value states。
#   在 use_cache=False + 单次 prefill 推理场景下完全等价。
# =============================================================================

def patch_sliding_window_cache() -> None:
    """
    将 transformers 里所有 *Cache 类的 update 方法替换为直通版本，
    彻底消除 index_copy_ / index_put_ → ScatterND。

    直通语义：update(k, v, ...) 直接返回 (k, v)，不写入任何 buffer。
    适用条件：use_cache=False 的单次 prefill 推理，与原语义等价。
    """
    import sys

    def _passthrough_update(self, key_states, value_states,
                            layer_idx, cache_kwargs=None):
        # 不做任何 scatter 写入，直接返回当前 step 的 kv
        return key_states, value_states

    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or "transformers" not in mod_name:
            continue
        for attr_name in dir(mod):
            try:
                cls = getattr(mod, attr_name)
            except Exception:
                continue
            if not isinstance(cls, type):
                continue
            # 匹配所有 Cache 类（SlidingWindowCache, StaticCache, DynamicCache 等）
            if "Cache" not in attr_name:
                continue
            if not hasattr(cls, "update"):
                continue
            # 只替换有实际写入操作的 update（检查源码含 index_copy_ 或 index_put）
            try:
                import inspect
                src = inspect.getsource(cls.update)
                has_scatter = any(op in src for op in (
                    "index_copy_", "index_put_", "index_put",
                    "scatter_", "scatter(", "+=",  # SlidingWindowCache 用 += 写回
                ))
            except Exception:
                has_scatter = True  # 无法获取源码时保守替换

            if has_scatter:
                cls.update = _passthrough_update
                patched.append(f"{mod_name}.{attr_name}")

    if patched:
        print(f"[patch_sliding_window_cache] patched: {patched}")
    else:
        print("[patch_sliding_window_cache] WARNING: no Cache classes found")

def patch_scatter(module):
    torch.Tensor.index_put_ = lambda self, indices, values, accumulate=False: self
    torch.Tensor.index_put  = lambda self, indices, values, accumulate=False: self.clone()
    torch.index_put         = lambda input, indices, values, accumulate=False: input.clone()
    torch.Tensor.masked_fill_ = lambda self, mask, value: self
    torch.Tensor.masked_fill  = lambda self, mask, value: self.clone()
    torch.Tensor.scatter_  = lambda self, dim, index, src, **kw: self
    torch.Tensor.scatter   = lambda self, dim, index, src, **kw: self.clone()

def apply_all_patches(model: nn.Module) -> nn.Module:
    # Cache classes already patched at import time by _patch_all_cache_classes()
    patch_rmsnorm(model)
    patch_rope(model)
    patch_apply_rotary(model)
    patch_attention(model)
    patch_causal_mask()
    patch_causal_mask_generation(model)
    patch_scatter(model)
    return model

# =============================================================================
# 模型配置
# =============================================================================

def _small_config(config):
    """
    在 from_config 之前设置所有可能产生 ScatterND 的配置项。
    关键：sliding_window=None 和 use_cache=False 必须在构建前设置。
    """
    config.use_cache = False                   # 禁用 KV cache
    config.rope_scaling = None                 # 禁用动态 NTK RoPE
    config._attn_implementation = "eager"      # 避免 flash/sdpa 路径
    config.num_hidden_layers = 4
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.max_position_embeddings = 64
    config.vocab_size = 256 * 256


def get_test_inputs(batch_size, config):
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask

# =============================================================================
# LLM 模型
# =============================================================================

def llama3(batch_size):
    config = AutoConfig.from_pretrained("openlm-research/open_llama_3b")
    _small_config(config)
    model = AutoModelForCausalLM.from_config(config)
    return apply_all_patches(model).eval(), get_test_inputs(batch_size, config)


def qwen2(batch_size):
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
    _small_config(config)
    config.sliding_window = None           # ★ from_config 前置 None
    model = AutoModelForCausalLM.from_config(config)
    return apply_all_patches(model).eval(), get_test_inputs(batch_size, config)


def mistral(batch_size):
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
    _small_config(config)
    config.sliding_window = None           # ★ 消除滑窗 cache（ScatterND 主因）
    if hasattr(config, "sliding_window_size"):
        config.sliding_window_size = None
    model = AutoModelForCausalLM.from_config(config)
    return apply_all_patches(model).eval(), get_test_inputs(batch_size, config)


def gemma(batch_size):
    config = AutoConfig.from_pretrained("alpindale/gemma-2b")
    _small_config(config)
    config.attn_logit_softcapping = None
    config.final_logit_softcapping = None
    config.query_pre_attn_scalar = None
    model = AutoModelForCausalLM.from_config(config)
    return apply_all_patches(model).eval(), get_test_inputs(batch_size, config)


def phi3(batch_size):
    config = AutoConfig.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    _small_config(config)
    config.original_max_position_embeddings = config.max_position_embeddings
    config.rope_scaling = None
    # ★ 确保 vocab_size 大于 padding_idx
    if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        config.vocab_size = max(config.vocab_size, config.pad_token_id + 1)
    model = AutoModelForCausalLM.from_config(config)
    return apply_all_patches(model).eval(), get_test_inputs(batch_size, config)