import subprocess
import os
import sys
os.environ["PATH"] = os.path.abspath(
    "/root/nnfusion/build/src/tools/nnfusion") + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath("/root/nnfusion/src/python"))
import argparse
import subprocess

from nnfusion.data_format import cast_numpy_array, cast_pytorch_tensor
from nnfusion.executor import Executor
from nnfusion.runtime import NNFusionRT
from nnfusion.session import PTSession as Session, generate_sample
from nnfusion.runner import PTRunner as Runner
from nnfusion.description import IODescription
from nnfusion.trainer import PTTrainer as Trainer
from nnfusion.utils import cd, execute
# fix_json.py
import json, re, sys
import onnx
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--index', type=int)
    parser.add_argument('--arch', type=str, default="A800")
    parser.add_argument("--no_tc", action="store_true", default=False)
    parser.add_argument("--welder_base", action="store_true", default=False)
    parser.add_argument("--welder_none", action="store_true", default=False)
    parser.add_argument("--skip_dot", action="store_true", default=False)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--prefix", type=str, default="./")
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    os.chdir(args.prefix)

    #os.chdir(args.prefix)
    path = ['bert','edgenext', 'efficientnet','mobilevit', 'NAFNet', 'NeRF','shufflenet', 'StarNet', 
 'swin_transformer', 'vgg11bn','llama3','qwen2','mistral','gemma','phi3']
    # for i in range(len(path)):
    i = args.index

    command0 = ["python", "./ModelTest/torch2onnx.py", str(path[i])]
    command1 = ["nnfusion", "./"+str(path[i])+"/model.onnx", "-f", "onnx", "-ftune_output_file=./"+str(path[i])+"/model.json"]
    command2 = ["python", "./run_compiler.py", '--input_file', "./"+str(path[i])+"/model.json", '--output_file', "./"+str(path[i])+"/output_"+path[i]+"_tunned.json",
                "--topk", str(args.topk), "--arch", args.arch, "--model_name", str(path[i]),"--epoch",str(args.epoch)]
    command3 = ["nnfusion", "./"+str(path[i])+"/model.onnx", "-f", "onnx", "-ftune_output_file=/dev/null", "-ftune_input_file=./"+str(path[i])+"/output_"+path[i]+"_tunned.json"]
    if args.no_tc:
        command1.append("-ftc_rewrite=0")
        command3.append("-ftc_rewrite=0")
    if args.welder_base:
        command2.append("--nofusion")
    elif args.welder_none:
        command1.append("-fnofuse=1")
        command2.append("--nofusion")
        command3.append("-fnofuse=1")
    if args.skip_dot:
        command1.append('-ffusion_skiplist=Dot')
        command3.append('-ffusion_skiplist=Dot')
    if args.bs!=1:
        command0.append("--bs")
        command0.append(str(args.bs))
    if args.fp16:
        command0.append("--fp16")
    #
    subprocess.run(command0, check=True)
    print("step1 is over")
    subprocess.run(command1, check=True)
    print("step2 is over")
    subprocess.run(command2, check=True)
    print("step3 is over")
    subprocess.run(command3, check=True)
    print("step4 is over")
    subprocess.run(["rm", "-rf", "./nnfusion_rt/cuda_codegen/build/"], check=True)
    print("step5 is over")
    subprocess.run(["cmake", "-S", "./nnfusion_rt/cuda_codegen/", "-B", "./res/"+str(path[i])+"/nnfusion_rt/cuda_codegen/build/"], check=True)
    print("step6 is over")
    subprocess.run(["make", "-C", "./res/"+str(path[i])+"/nnfusion_rt/cuda_codegen/build/"], check=True)
    print("step7 is over")
    # subprocess.run(['cp', './build/libnnfusion_naive_rt.so', './'], check=True)
    # rt_dir =  "/home/xiarui/ATF4NN_New/nnfusion_rt/cuda_codegen/"
    # executor = Executor(rt_dir)
    # input_name = executor.get_inputs()[0].name
    # output_name = executor.get_outputs()[0].name

    #print(input_name,output_name)
