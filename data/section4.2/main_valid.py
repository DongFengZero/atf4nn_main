import numpy as np

models = ['Bert', 'EdgeNeXt', 'MobileViT', 'NAFNet', 'NeRF', 'ShuffleNet', 'StarNet', 'Swin Transformer']
methods = ['PyTorch', 'OnnxRuntime', 'TensorRT', 'Welder', 'Ours']
MI = {m: i for i, m in enumerate(methods)}
MDI = {m: i for i, m in enumerate(models)}

all_data = {
    'RTX 4090': {
        64: [
            [43.63, 32.79, 21.40, 36.16, 36.05],
            [6.22,  5.68,  2.94,  2.00,  1.82],
            [13.35, 14.15, 8.07,  4.25,  3.51],
            [82.25, 93.79, 50.97, 16.65, 15.49],
            [26.36, 27.26, 14.98, 6.27,  6.22],
            [2.84,  2.90,  2.50,  1.64,  1.58],
            [7.30,  7.94,  4.48,  3.24,  3.17],
            [131.05,136.54,76.42, 85.22, 84.60],
        ],
        96: [
            [68.04, 51.90, 34.91, 55.97, 55.41],
            [10.19, 8.38,  4.50,  2.67,  2.76],
            [20.94, 22.61, 12.67, 6.26,  5.16],
            [124.96,142.47,76.67, 24.75, 22.99],
            [39.50, 40.91, 22.44, 9.39,  9.31],
            [4.53,  4.05,  3.68,  2.52,  2.40],
            [11.60, 12.61, 6.81,  4.64,  4.56],
            [200.86, None, 116.51,131.39,129.76],
        ],
        128: [
            [91.73, 71.73, 45.31, 73.31, 72.58],
            [12.04, 11.46, 6.56,  3.53,  3.35],
            [28.48, 31.52, 17.53, 8.24,  6.86],
            [167.69,190.41,102.63,33.13, 30.73],
            [52.65, 54.52, 30.03, 12.51, 12.43],
            [6.08,  5.29,  4.82,  3.19,  3.02],
            [16.10, 17.91, 9.40,  6.26,  6.07],
            [272.51, None, 156.08,175.88,173.50],
        ],
    },
    'RTX 3090 Ti': {
        64: [
            [77.97, 57.21, 41.49, 66.30, 65.37],
            [12.23, 11.12, 5.62,  3.46,  3.44],
            [20.01, 20.89, 11.07, 7.32,  6.55],
            [90.07, 103.41,53.93, 22.28, 20.06],
            [28.13, 35.15, 15.08, 11.55, 11.10],
            [5.79,  4.62,  4.62,  3.17,  3.10],
            [12.77, 14.00, 7.47,  6.21,  6.08],
            [213.04,200.65,113.12,148.03,140.86],
        ],
        96: [
            [114.80,84.40, 60.54, 96.46, 95.69],
            [17.94, 15.97, 8.28,  4.67,  5.42],
            [29.47, 30.66, 16.15, 10.58, 9.25],
            [134.59,153.82,80.40, 33.15, 29.88],
            [42.10, 52.65, 22.60, 18.17, 16.59],
            [8.40,  6.68,  6.32,  4.52,  4.31],
            [18.81, 21.33, 10.64, 8.39,  8.84],
            [316.63, None, 167.46,213.47,204.17],
        ],
        128: [
            [153.16,111.98,81.97, 128.20,127.31],
            [23.55, 20.87, 10.87, 6.36,  6.30],
            [38.79, 40.46, 21.25, 13.90, 12.56],
            [179.18,204.10,107.06,44.08, 40.05],
            [56.03, 70.06, 30.10, 23.10, 22.16],
            [11.19, 8.62,  8.24,  5.75,  5.69],
            [24.73, 27.68, 13.87, 11.37, 11.29],
            [418.99, None, 221.93,282.40,270.22],
        ],
    },
    'A800': {
        64: [
            [94.81, 28.25, 18.46, 87.08, 86.64],
            [13.02, 11.62, 4.02,  3.33,  3.29],
            [14.92, 17.22, 8.26,  6.55,  5.64],
            [58.59, 78.06, 37.14, 17.01, 15.43],
            [21.34, 15.53, 8.82,  13.13, 13.01],
            [8.83,  6.20,  4.43,  2.98,  2.71],
            [8.91,  16.58, 6.20,  5.80,  5.62],
            [214.00,130.23,60.82, 174.66,169.51],
        ],
        96: [
            [134.15,41.51, 26.33, 124.77,124.64],
            [16.17, 16.19, 5.85,  4.59,  4.55],
            [21.85, 25.00, 11.93, 9.82,  8.22],
            [87.41, 116.05,54.92, 24.68, 22.77],
            [31.62, 23.16, 13.21, 19.32, 19.08],
            [9.00,  8.67,  6.06,  4.16,  3.98],
            [13.14, 23.90, 8.59,  7.94,  8.09],
            [312.20, None, 87.71, 263.79,252.06],
        ],
        128: [
            [180.24,54.71, 34.92, 170.08,168.88],
            [20.55, 17.76, 7.47,  5.90,  5.71],
            [28.66, 31.65, 15.47, None,  10.69],
            [115.93,154.40,72.72, 33.28, 30.17],
            [42.09, 30.98, 17.55, 25.96, 25.50],
            [8.93,  7.96,  7.29,  5.29,  4.85],
            [17.08, 21.62, 10.96, 10.45, 10.45],
            [410.00, None, 115.21,352.85,331.52],
        ],
    },
}

def get(gpu, bs, model, method):
    return all_data[gpu][bs][MDI[model]][MI[method]]

def improvement(welder, ours):
    """Ours vs Welder: positive = Ours faster (lower latency)"""
    if welder is None or ours is None:
        return None
    return (welder - ours) / welder * 100

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def check(label, computed, claimed, tol=0.01):
    ok = abs(computed - claimed) <= tol
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    print(f"          claimed={claimed:.2f}  computed={computed:.4f}  diff={computed-claimed:+.4f}")

def check_none(label, val):
    ok = val is None
    status = PASS if ok else FAIL
    print(f"  {status}  {label}  (value={'None' if val is None else val})")

print("=" * 70)
print("SECTION 1 — A800")
print("=" * 70)

print("\n[A800] MobileViT bs=128: Welder failed (None), Ours=2.68× over PyTorch")
check_none("Welder MobileViT bs=128 is None", get('A800', 128, 'MobileViT', 'Welder'))
pytorch_val = get('A800', 128, 'MobileViT', 'PyTorch')
ours_val    = get('A800', 128, 'MobileViT', 'Ours')
check("MobileViT bs=128 speedup vs PyTorch = 2.68×", pytorch_val / ours_val, 2.68)

print("\n[A800] bs=128 most significant improvements: NAFNet 9.34%, ShuffleNet 8.32%, Swin 6.05%")
check("NAFNet    bs=128  Ours vs Welder = 9.34%",
      improvement(get('A800',128,'NAFNet','Welder'),       get('A800',128,'NAFNet','Ours')),       9.34)
check("ShuffleNet bs=128 Ours vs Welder = 8.32%",
      improvement(get('A800',128,'ShuffleNet','Welder'),   get('A800',128,'ShuffleNet','Ours')),   8.32)
check("Swin      bs=128  Ours vs Welder = 6.05%",
      improvement(get('A800',128,'Swin Transformer','Welder'), get('A800',128,'Swin Transformer','Ours')), 6.05)

print("\n[A800] StarNet: improvement at bs=64, comparable to Welder at bs=128")
imp64  = improvement(get('A800',64, 'StarNet','Welder'), get('A800',64, 'StarNet','Ours'))
imp96  = improvement(get('A800',96, 'StarNet','Welder'), get('A800',96, 'StarNet','Ours'))
imp128 = improvement(get('A800',128,'StarNet','Welder'), get('A800',128,'StarNet','Ours'))
print(f"  StarNet Ours vs Welder — bs=64: {imp64:+.2f}%  bs=96: {imp96:+.2f}%  bs=128: {imp128:+.2f}%")
print(f"  {'✅' if imp64 > 0 else '❌'}  bs=64 Ours wins (>0%)")
print(f"  {'✅' if abs(imp128) < 0.01 else '❌'}  bs=128 essentially tied (≈0%)")

print("\n" + "=" * 70)
print("SECTION 2 — RTX 3090 Ti")
print("=" * 70)

print("\n[3090Ti] bs=128: MobileViT 9.64%, NAFNet 9.14%, Swin 4.31%")
check("MobileViT  bs=128 Ours vs Welder = 9.64%",
      improvement(get('RTX 3090 Ti',128,'MobileViT','Welder'),        get('RTX 3090 Ti',128,'MobileViT','Ours')),        9.64)
check("NAFNet     bs=128 Ours vs Welder = 9.14%",
      improvement(get('RTX 3090 Ti',128,'NAFNet','Welder'),           get('RTX 3090 Ti',128,'NAFNet','Ours')),           9.14)
check("Swin       bs=128 Ours vs Welder = 4.31%",
      improvement(get('RTX 3090 Ti',128,'Swin Transformer','Welder'), get('RTX 3090 Ti',128,'Swin Transformer','Ours')), 4.31)

print("\n[3090Ti] EdgeNeXt: moderate speedup at bs=64 and bs=128")
for bs in [64, 96, 128]:
    imp = improvement(get('RTX 3090 Ti',bs,'EdgeNeXt','Welder'), get('RTX 3090 Ti',bs,'EdgeNeXt','Ours'))
    print(f"  EdgeNeXt bs={bs}: Ours vs Welder = {imp:+.2f}%")

print("\n[3090Ti] StarNet: moderate speedup at bs=64 and bs=128")
for bs in [64, 96, 128]:
    imp = improvement(get('RTX 3090 Ti',bs,'StarNet','Welder'), get('RTX 3090 Ti',bs,'StarNet','Ours'))
    print(f"  StarNet  bs={bs}: Ours vs Welder = {imp:+.2f}%")

print("\n" + "=" * 70)
print("SECTION 3 — RTX 4090")
print("=" * 70)

print("\n[4090] bs=128: MobileViT 16.75%, NAFNet 7.24%")
check("MobileViT bs=128 Ours vs Welder = 16.75%",
      improvement(get('RTX 4090',128,'MobileViT','Welder'), get('RTX 4090',128,'MobileViT','Ours')), 16.75)
check("NAFNet    bs=128 Ours vs Welder = 7.24%",
      improvement(get('RTX 4090',128,'NAFNet','Welder'),    get('RTX 4090',128,'NAFNet','Ours')),    7.24)

print("\n[4090] EdgeNeXt: moderate speedup at bs=64 and bs=128")
for bs in [64, 96, 128]:
    imp = improvement(get('RTX 4090',bs,'EdgeNeXt','Welder'), get('RTX 4090',bs,'EdgeNeXt','Ours'))
    print(f"  EdgeNeXt bs={bs}: Ours vs Welder = {imp:+.2f}%")

print("\n" + "=" * 70)
print("SECTION 4 — ONNXRuntime & TensorRT average speedups")
print("=" * 70)

for gpu, claimed_onnx, claimed_trt in [
    ('RTX 4090',    3.28, 1.81),
    ('RTX 3090 Ti', 2.72, 1.44),
    ('A800',        2.45, 1.13),
]:
    onnx_ratios, trt_ratios = [], []
    for bs in [64, 96, 128]:
        for mi, model in enumerate(models):
            row = all_data[gpu][bs][mi]
            pytorch, onnx, trt, welder, ours = row
            if onnx is not None and ours is not None:
                onnx_ratios.append(onnx / ours)
            if trt is not None and ours is not None:
                trt_ratios.append(trt / ours)
    avg_onnx = np.mean(onnx_ratios)
    avg_trt  = np.mean(trt_ratios)
    print(f"\n  {gpu}")
    check(f"avg speedup vs ONNXRuntime = {claimed_onnx:.2f}×", avg_onnx, claimed_onnx)
    check(f"avg speedup vs TensorRT    = {claimed_trt:.2f}×",  avg_trt,  claimed_trt)

print("\n" + "=" * 70)
print("SECTION 5 — Rank-1 and Top-2 statistics")
print("=" * 70)

total = rank1 = top2 = 0
model_rank1 = {m: 0 for m in models}
model_total = {m: 0 for m in models}

for gpu in ['RTX 4090', 'RTX 3090 Ti', 'A800']:
    for bs in [64, 96, 128]:
        for mi, model in enumerate(models):
            row = all_data[gpu][bs][mi]
            valid = [(v, i) for i, v in enumerate(row) if v is not None]
            if not valid:
                continue
            sorted_v = sorted(valid, key=lambda x: x[0])
            best = sorted_v[0][0]
            second = sorted_v[1][0] if len(sorted_v) > 1 else None
            ours = row[MI['Ours']]
            total += 1
            model_total[model] += 1
            if ours is not None:
                if abs(ours - best) < 1e-9:
                    rank1 += 1
                    model_rank1[model] += 1
                if second is not None and ours <= second + 1e-9:
                    top2 += 1

rank1_pct = rank1 / total * 100
top2_pct  = top2  / total * 100

print(f"\n  Total scenarios : {total}")
print(f"  Rank-1 count    : {rank1}  →  {rank1_pct:.1f}%")
print(f"  Top-2  count    : {top2}  →  {top2_pct:.1f}%")

# Claimed: 65.3% rank-1, 86.1% top-2
check("Rank-1 = 65.3%", rank1_pct, 65.3, tol=0.15)   # 允许0.15个百分点容差
check("Top-2  = 86.1%", top2_pct,  86.1, tol=0.05)

print("\n  Per-model rank-1 (paper claims MobileViT / NAFNet / ShuffleNet all 9/9):")
for m in models:
    mark = "✅" if model_rank1[m] == model_total[m] else "  "
    print(f"  {mark}  {m:20s}: {model_rank1[m]}/{model_total[m]}")

for m in ['MobileViT', 'NAFNet', 'ShuffleNet']:
    ok = model_rank1[m] == 9
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {m} rank-1 in all 9 scenarios")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)