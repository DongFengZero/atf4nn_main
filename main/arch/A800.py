import tvm


class A800:
    def __init__(self):
        self.reg_cap = 64 * 1024
        self.smem_cap = 2 * 41 * 1024
        self.compute_max_core = 108
        self.warp_size = 32
        self.sm_partition = 4

        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 164 * 1024

        self.bandwidth = [1935, 19035]
        self.platform = "CUDA"
        self.L2_size = 40960 * 1024
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A800", arch="sm_80")
        self.cutlass_mma = [16, 8, 16]