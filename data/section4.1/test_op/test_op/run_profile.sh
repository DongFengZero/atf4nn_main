#!/usr/bin/env bash
# =============================================================================
# run_profile.sh  —  flash_attn DRAM 访存 Nsight Compute 采集脚本
# =============================================================================
#
# 测试配置：
#   算子  : flash_attn（3 kernels：tiled_gemm_h → row_softmax_h → tiled_gemm_h）
#   Tile  : S = 768 - 2048
#   组数  : 100 组，每组 3 次 kernel 启动，共 300 次
#   ncu   : --launch-count 300，捕获全部 300 次启动
#
# CSV ID 编码规则（供 plot_dram.py 解析）：
#   rep r，kernel k  →  ID = r*3 + k
#   k=0  tiled_gemm_h   K0: Q×K^T → S'
#   k=1  row_softmax_h  K1: softmax in-place
#   k=2  tiled_gemm_h   K2: S'×V  → O
#
# 输出: <RESULT_DIR>/flash_attn_S<tile>.csv
#
# 用法:
#   chmod +x run_profile.sh
#   ./run_profile.sh                          # 使用默认参数
#   ./run_profile.sh ./benchmark ./results    # 指定路径
#   REPS=50 ./run_profile.sh                  # 修改重复次数
#
# 前置条件:
#   make                                      # 编译 benchmark
#   sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
# =============================================================================

set -euo pipefail

# ── 参数 ─────────────────────────────────────────────────────
BENCH="${1:-./benchmark/benchmark}"
RESULT_ROOT="${2:-./results}"

TILES=(768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048)
REPS="${REPS:-100}"
NKERNELS=3
LAUNCH_CNT=$(( REPS * NKERNELS ))    # 300

# Nsight Compute >= 2021.1 使用双下划线格式：
#   dram__bytes_read.sum  /  dram__bytes_write.sum
# 旧版（< 2021.1）使用：
#   dram_read_bytes  /  dram_write_bytes
# 此处自动探测，优先新版格式
if ncu --query-metrics 2>/dev/null | grep -q 'dram__bytes_read'; then
    NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum"
else
    NCU_METRICS="dram_read_bytes,dram_write_bytes"
fi

# ── 依赖检查 ─────────────────────────────────────────────────
if ! command -v ncu &>/dev/null; then
    echo "[ERROR] ncu not found. Add Nsight Compute to PATH." >&2
    exit 1
fi
if [[ ! -x "$BENCH" ]]; then
    echo "[ERROR] $BENCH not found or not executable. Run 'make' first." >&2
    exit 1
fi

# ── 输出目录（按 GPU 型号命名）────────────────────────────────
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
          | head -1 | sed 's/ /_/g; s/[^A-Za-z0-9_-]//g')
GPU_TAG="${GPU_TAG:-unknown_gpu}"
RESULT_DIR="${RESULT_ROOT}/${GPU_TAG}"
mkdir -p "$RESULT_DIR"

# ── 打印配置 ─────────────────────────────────────────────────
echo "============================================================"
printf " %-12s %s\n" "op"        "flash_attn"
printf " %-12s %s\n" "tiles"     "${TILES[*]}"
printf " %-12s %s\n" "reps"      "${REPS}  (x${NKERNELS} kernels = ${LAUNCH_CNT} launches)"
printf " %-12s %s\n" "metrics"   "${NCU_METRICS}"
printf " %-12s %s\n" "output"    "${RESULT_DIR}"
echo "============================================================"

# ── 采集循环 ─────────────────────────────────────────────────
for S in "${TILES[@]}"; do
    TAG=$(printf 'S%04d' "$S")
    OUT="${RESULT_DIR}/flash_attn_${TAG}.csv"

    echo ""
    echo ">>> S=${S}  ->  ${OUT}"

    ncu \
        --csv \
        --metrics      "${NCU_METRICS}" \
        --launch-count "${LAUNCH_CNT}" \
        --target-processes all \
        "${BENCH}" --op flash_attn --tile "${S}" --reps "${REPS}" \
        > "${OUT}" 2>/dev/null

    LINES=$(wc -l < "${OUT}")
    # 期望行数：2 行 ncu 头 + LAUNCH_CNT * 2 指标行 = 602 行
    EXPECTED=$(( 2 + LAUNCH_CNT * 2 ))
    if (( LINES >= EXPECTED - 10 )); then
        echo "    OK  (${LINES} lines)"
    else
        echo "    [WARN] Only ${LINES} lines (expected ~${EXPECTED})."
        echo "    Fix: sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'"
    fi
done

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Done. Results: ${RESULT_DIR}/"
ls -lh "${RESULT_DIR}"/flash_attn_S*.csv 2>/dev/null || true
echo ""
echo " Next: python3 plot_dram.py ${RESULT_DIR}"
echo "============================================================"
