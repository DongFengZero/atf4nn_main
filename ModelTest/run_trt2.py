import tensorrt as trt
import numpy as np
import os.path as osp
import time
import argparse
import torch
def run_trt(prefix, use_fp16=False):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(osp.join(prefix, "model.onnx"), 'rb') as f:
        success = parser.parse(f.read())
    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    # ── 修复1: STRICT_TYPES 在 TensorRT 10.x 中已移除 ──
    # 旧: config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    # 新: 使用 OBEY_PRECISION_CONSTRAINTS（如果需要严格精度控制）
    #config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # ── 修复2: build_engine() 在 TensorRT 10.x 中已移除 ──
    # 旧: engine = builder.build_engine(network, config)
    # 新: 先 build_serialized_network，再 deserialize
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("Built engine successfully.")
    # ── 修复3: get_binding_shape / get_binding_dtype 在 TensorRT 10.x 中已移除 ──
    # 旧: engine[index], engine.get_binding_shape(), engine.get_binding_dtype()
    # 新: 使用 engine.get_tensor_name / get_tensor_shape / get_tensor_dtype
    num_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
    tensors = {}
    for name in tensor_names:
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        # trt.DataType -> numpy dtype 映射
        np_dtype = trt.nptype(dtype)
        torch_dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
        tensors[name] = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
    # 加载输入数据并覆盖对应的输入 tensor
    feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))
    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT and name in feed_dict:
            tensors[name] = torch.from_numpy(feed_dict[name]).cuda()
    context = engine.create_execution_context()
    # ── 修复4: execute_v2(bindinglist) 在 TensorRT 10.x 中已移除 ──
    # 旧: context.execute_v2([t.data_ptr() for t in tensors])
    # 新: 使用 set_tensor_address + execute_async_v3
    stream = torch.cuda.Stream()
    for name in tensor_names:
        context.set_tensor_address(name, tensors[name].data_ptr())
    def get_runtime():
        tic = time.monotonic_ns()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return (time.monotonic_ns() - tic) / 1e6
    print("Warming up ...")
    st = time.time()
    while time.time() - st < 1.0:
        get_runtime()
    times = [get_runtime() for _ in range(100)]
    print(f"avg: {np.mean(times):.3f} ms")
    print(f"min: {np.min(times):.3f} ms")
    print(f"max: {np.max(times):.3f} ms")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    torch.random.manual_seed(0)
    run_trt(args.prefix, args.fp16)