import tvm


class g4090:
    def __init__(self):
        self.reg_cap = 64 * 1024
        self.smem_cap = 25 * 1024
        self.compute_max_core = 128
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 100 * 1024

        self.bandwidth = [1008, 35760]
        self.platform = "CUDA"
        self.L2_size = 73728 * 1024
        self.compute_capability = "89"
        self.target = tvm.target.cuda(model="g4090", arch="sm_89")
        self.cutlass_mma = [16, 8, 16]