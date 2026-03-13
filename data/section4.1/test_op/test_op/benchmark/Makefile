# Makefile for benchmark.cu  —  no external libraries required

NVCC    ?= nvcc
NCU     ?= ncu
TARGET   = benchmark
SRC      = benchmark.cu

# ── Auto-detect GPU compute capability ──────────────────────
DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
              2>/dev/null | head -1 | tr -d '.')

ifeq ($(DETECTED),)
    ARCH = sm_80
    $(warning Cannot detect GPU, defaulting to sm_80)
else
    ARCH = sm_$(DETECTED)
    $(info GPU compute cap: $(DETECTED) -> -arch=$(ARCH))
endif

# ── Profile parameters ───────────────────────────────────────
# Tiles: S = 640..2048 step 128  (12 points)
# Reps:  100 groups, each group = 3 kernel launches
# ncu captures all 300 launches per tile (--launch-count 300)
TILES      = 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048
REPS      ?= 100
NKERNELS   = 3
LAUNCH_CNT = $(shell expr $(REPS) \* $(NKERNELS))

NCU_METRICS = dram_read_bytes,dram_write_bytes

GPU_TAG    := $(shell nvidia-smi --query-gpu=name --format=csv,noheader \
                2>/dev/null | head -1 | sed 's/ /_/g; s/[^A-Za-z0-9_-]//g')
RESULT_DIR ?= results/$(GPU_TAG)

# ── Targets ──────────────────────────────────────────────────
.PHONY: all sm80 sm86 sm89 test profile plot clean cleanresults

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -o $@ $<
	@echo "Build OK: ./$(TARGET)"

sm80: ; $(NVCC) -O2 -std=c++17 -arch=sm_80 -o $(TARGET) $(SRC)
sm86: ; $(NVCC) -O2 -std=c++17 -arch=sm_86 -o $(TARGET) $(SRC)
sm89: ; $(NVCC) -O2 -std=c++17 -arch=sm_89 -o $(TARGET) $(SRC)

# ── Smoke test ───────────────────────────────────────────────
test: $(TARGET)
	@echo "=== smoke test ==="
	./$(TARGET) --op flash_attn --tile 256 --reps 2
	./$(TARGET) --op cbr        --tile 256 --reps 2
	./$(TARGET) --op layer_norm --tile 256 --reps 2
	@echo "=== all passed ==="

# ── Nsight Compute profile ───────────────────────────────────
# Profiles flash_attn only across S=640..1024.
# Each tile runs benchmark with --reps $(REPS).
# That produces REPS*3 kernel launches; ncu --launch-count captures all.
#
# CSV ID encoding:
#   rep r, kernel k  ->  ID = r*3 + k
#   k=0  tiled_gemm_h  (K0: Q x K^T)
#   k=1  row_softmax_h (K1: softmax)
#   k=2  tiled_gemm_h  (K2: S' x V)
#
# Prerequisites:
#   sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
#
# Overrides:
#   make profile REPS=50
#   make profile RESULT_DIR=results/run2
profile: $(TARGET)
	@mkdir -p $(RESULT_DIR)
	@echo "============================================================"
	@echo " op         : flash_attn"
	@echo " tiles      : $(TILES)"
	@echo " reps       : $(REPS)  (x$(NKERNELS) kernels = $(LAUNCH_CNT) launches)"
	@echo " metrics    : $(NCU_METRICS)"
	@echo " output dir : $(RESULT_DIR)"
	@echo "============================================================"
	@for S in $(TILES); do \
	    OUT=$(RESULT_DIR)/flash_attn_S$$(printf '%04d' $$S).csv; \
	    echo ""; \
	    echo ">>> S=$$S  ->  $$OUT"; \
	    $(NCU) \
	        --csv \
	        --metrics "$(NCU_METRICS)" \
	        --launch-count $(LAUNCH_CNT) \
	        --target-processes all \
	        ./$(TARGET) --op flash_attn --tile $$S --reps $(REPS) \
	        > $$OUT 2>/dev/null; \
	    echo "    $$(wc -l < $$OUT) lines written"; \
	done
	@echo ""
	@echo "============================================================"
	@echo " Done. Results: $(RESULT_DIR)/"
	@echo " Next: python3 plot_dram.py $(RESULT_DIR)"
	@echo "============================================================"

# ── Plot (after profiling) ────────────────────────────────────
plot:
	python3 plot_dram.py $(RESULT_DIR)

# ── Cleanup ──────────────────────────────────────────────────
clean:
	rm -f $(TARGET)

cleanresults:
	rm -rf results/
