class Arch:
    def __init__(self) -> None:
        self.reg_cap = 0
        self.smem_cap = 0
        self.compute_max_core = 0
        self.warp_size = 0
        self.sm_partition = 0
        self.transaction_size = [0, 0]
        self.capacity = 0
        self.max_smem_usage = 0
        self.block_per_sm = 8
        self.ratio = 0
        self.ratio2 = 0
        self.L2_size = 0
        self.bandwidth = [0, 0]
        self.platform = "unknown"
        self.compute_capability = "unknown"
