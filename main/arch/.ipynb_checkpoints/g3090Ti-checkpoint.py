import tvm


class g3090Ti:
    def __init__(self):
        self.reg_cap = 65536
        self.smem_cap = 25*1024
        self.compute_max_core = 84
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 100 * 1024
        self.bandwidth = [1008, 19999]
        self.L2_size = 6144 * 1024
        self.platform = "CUDA"
        self.compute_capability = "86"
        self.target = tvm.target.cuda(model="g3090Ti", arch="sm_86")

        self.cutlass_mma = [16, 8, 16]
