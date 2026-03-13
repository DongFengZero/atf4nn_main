import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ── Data ──────────────────────────────────────────────────────────────────────
models = ['Bert', 'EdgeNeXt', 'MobileViT', 'NAFNet', 'NeRF', 'ShuffleNet', 'StarNet', 'Swin\nTransformer']
methods = ['PyTorch', 'OnnxRuntime', 'TensorRT', 'Welder', 'Ours']

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

def normalize(data):
    normed = []
    for row in data:
        pytorch_val = row[0]
        normed.append([None if v is None else v / pytorch_val for v in row])
    return normed

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

COLORS = {
    'PyTorch':     '#4C72B0',
    'OnnxRuntime': '#DD8452',
    'TensorRT':    '#55A868',
    'Welder':      '#C44E52',
    'Ours':        '#8172B2',
}
HATCH = {
    'PyTorch':     '',
    'OnnxRuntime': '////',
    'TensorRT':    '\\\\\\\\',
    'Welder':      '....',
    'Ours':        'xxxx',
}

batch_sizes = [64, 96, 128]
n_models  = len(models)
n_methods = len(methods)
bar_w     = 0.14
group_gap = 0.08
x_scale   = 0.82

# ── Font sizes ─────────────────────────────────────────────────────────────────
# Figure will be ~16×5 inches. After LaTeX scales it to \linewidth (~6.5 in),
# the effective DPI means these sizes render clearly in the final PDF.
FS_SUPTITLE    = 13
FS_SUBTITLE    = 11
FS_XTICKLABEL  = 9.5
FS_YTICKLABEL  = 9
FS_YLABEL      = 10
FS_BAR_BEST    = 8.5
FS_BAR_NORMAL  = 7.5
FS_CROSS       = 13
FS_LEGEND      = 9.5
FS_LEGEND_TTL  = 10


def draw_gpu_figure(gpu_name, gpu_raw):
    norm = {bs: normalize(gpu_raw[bs]) for bs in batch_sizes}

    # Compact canvas — high font-to-figure ratio survives LaTeX downscaling
    fig = plt.figure(figsize=(18, 5.2), facecolor='#F8F9FA')
    fig.suptitle(
        f'Normalized Inference Time (PyTorch = 1.0)  |  Lower is Better  |  GPU: {gpu_name}',
        fontsize=FS_SUPTITLE, fontweight='bold', y=1.02, color='#1a1a2e'
    )
    gs = GridSpec(1, 3, figure=fig, wspace=0.28,
                  left=0.06, right=0.98, top=0.88, bottom=0.28)

    for bi, bs in enumerate(batch_sizes):
        ax = fig.add_subplot(gs[0, bi])
        ax.set_facecolor('#FFFFFF')

        data      = norm[bs]
        x_centers = np.arange(n_models) * x_scale
        total_w   = n_methods * bar_w + group_gap
        offsets   = np.linspace(-(n_methods-1)/2 * bar_w,
                                 (n_methods-1)/2 * bar_w, n_methods)

        best_per_model = []
        for ri in range(n_models):
            valid = [data[ri][mi] for mi in range(n_methods) if data[ri][mi] is not None]
            best_per_model.append(min(valid) if valid else None)

        for mi, method in enumerate(methods):
            vals = [data[ri][mi] for ri in range(n_models)]
            xs   = x_centers + offsets[mi]
            for ri, (x, v) in enumerate(zip(xs, vals)):
                if v is None:
                    ax.text(x, 0.05, '✗', ha='center', va='bottom',
                            fontsize=FS_CROSS, color='#CC3333', fontweight='bold')
                    continue
                ax.bar(x, v, width=bar_w * 0.88,
                       color=COLORS[method], alpha=0.88,
                       hatch=HATCH[method], edgecolor='white', linewidth=0.5)
                if method != 'PyTorch':
                    is_best = (best_per_model[ri] is not None and
                               abs(v - best_per_model[ri]) < 1e-9)
                    ax.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom',
                            fontsize=FS_BAR_BEST if is_best else FS_BAR_NORMAL,
                            color='#000' if is_best else '#555',
                            rotation=90,
                            fontweight='bold' if is_best else 'normal')

        ax.axhline(1.0, color='#4C72B0', linewidth=1.6, linestyle='--', alpha=0.7)

        ax.set_title(f'Batch Size = {bs}', fontsize=FS_SUBTITLE, fontweight='bold',
                     color='#1a1a2e', pad=7)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(models, fontsize=FS_XTICKLABEL, rotation=35, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='y', labelsize=FS_YTICKLABEL)
        ax.set_ylabel('Normalized Time', fontsize=FS_YLABEL)
        ax.set_xlim(x_centers[0] - total_w/2 - 0.06,
                    x_centers[-1] + total_w/2 + 0.06)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.40)

        for xi, xc in enumerate(x_centers):
            if xi % 2 == 0:
                ax.axvspan(xc - total_w/2, xc + total_w/2,
                           alpha=0.04, color='#888', zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=COLORS[m], hatch=HATCH[m],
                       edgecolor='#555', label=m, alpha=0.88)
        for m in methods
    ]
    fail_marker = Line2D([0], [0], marker='x', color='#CC3333',
                         label='Not Applicable',
                         markersize=9, linewidth=0, markeredgewidth=2.2)
    legend_patches.append(fail_marker)
    fig.legend(handles=legend_patches, loc='lower center', ncol=6,
               fontsize=FS_LEGEND, frameon=True, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.04),
               title='Inference Framework', title_fontsize=FS_LEGEND_TTL,
               handlelength=2.0, handleheight=1.4)

    return fig


# ── Generate PDFs ──────────────────────────────────────────────────────────────
import os
os.makedirs('./outputs1', exist_ok=True)

gpu_files = {
    'RTX 4090':    './outputs1/inference_g4090.pdf',
    'RTX 3090 Ti': './outputs1/inference_g3090Ti.pdf',
    'A800':        './outputs1/inference_A800.pdf',
}

for gpu_name, fpath in gpu_files.items():
    fig = draw_gpu_figure(gpu_name, all_data[gpu_name])
    with PdfPages(fpath) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {fpath}")

# ── LaTeX caption using minipage ───────────────────────────────────────────────
latex = r"""\begin{figure}[p]
  \centering

  \begin{minipage}{\linewidth}
    \centering
    \includegraphics[width=\linewidth]{inference_g4090.pdf}
    \subcaption{NVIDIA RTX 4090}
    \label{fig:inference_g4090}
  \end{minipage}

  \vspace{0.6em}

  \begin{minipage}{\linewidth}
    \centering
    \includegraphics[width=\linewidth]{inference_g3090Ti.pdf}
    \subcaption{NVIDIA RTX 3090 Ti}
    \label{fig:inference_g3090Ti}
  \end{minipage}

  \vspace{0.6em}

  \begin{minipage}{\linewidth}
    \centering
    \includegraphics[width=\linewidth]{inference_A800.pdf}
    \subcaption{NVIDIA A800}
    \label{fig:inference_A800}
  \end{minipage}

  \caption{Normalized single-batch inference latency across three GPUs (RTX~4090,
    RTX~3090~Ti, and A800) and three batch sizes (64, 96, 128). All values are
    normalized to PyTorch (\emph{i.e.}, PyTorch~$= 1.0$); lower is better.
    \textbf{Bold} labels above bars indicate the best-performing framework for
    each model. \textcolor{red}{$\times$} indicates that the framework failed
    to execute the corresponding model (out-of-memory or runtime crash).
    Results cover eight representative architectures spanning transformers,
    lightweight CNNs, and implicit neural representations.}
  \label{fig:inference_latency_all}
\end{figure}

% Required packages in preamble:
%   \usepackage{subcaption}
%   \usepackage{graphicx}
%   \usepackage{xcolor}
% Use [p] float specifier so LaTeX places this on a dedicated float page."""

with open('./outputs1/latex_minipage.txt', 'w') as f:
    f.write(latex)
print("Saved: latex_minipage.txt")