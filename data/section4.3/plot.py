"""
plot_fusion_acm.py  —  ACM two-column combined figure
======================================================
One PDF containing three sub-figures (a) A800  (b) g3090Ti  (c) g4090,
each with 4 rows × 2 cols of model panels.  A single shared legend sits
at the bottom (≤ 2 rows).  Sub-figure labels (a)(b)(c) appear below
each column title, matching ACM subfigure convention.

Target dimensions (ACM two-column, \textwidth = 6.50 in):
  Total width   : 6.50 in  (full text width)
  Total height  : 7.80 in  (4 panel rows + title + legend)
  Per column    : ≈ 2.10 in wide  (3 cols × 2.10 + gutters ≈ 6.50 in)
  Fonts         : Times New Roman + STIX math
  Output        : PDF (Type-42) + PNG 600 dpi

LaTeX usage (single figure, one caption):
  \begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fusion_gain_combined.pdf}
    \caption{Fusion latency improvement (ms) across RL convergence epochs
      on (a) NVIDIA A800, (b) NVIDIA RTX 3090 Ti, (c) NVIDIA RTX 4090
      (batch size = 128).}
    \label{fig:reward_all}
  \end{figure*}
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import glob, os

# ── ACM rcParams ───────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size'          : 7.0,
    'mathtext.fontset'   : 'stix',
    'text.usetex'        : False,
    'axes.linewidth'     : 0.60,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.unicode_minus' : False,
    'xtick.direction'    : 'out', 'ytick.direction'    : 'out',
    'xtick.major.size'   : 2.2,   'ytick.major.size'   : 2.2,
    'xtick.major.width'  : 0.50,  'ytick.major.width'  : 0.50,
    'xtick.labelsize'    : 6.0,   'ytick.labelsize'    : 6.0,
    'grid.alpha'         : 0.25,
    'legend.framealpha'  : 0.97,
    'legend.edgecolor'   : '#444444',
    'legend.fontsize'    : 6.5,
    'legend.handlelength': 1.6,
    'legend.borderpad'   : 0.40,
    'legend.labelspacing': 0.22,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 600,
    'pdf.fonttype'       : 42,
    'ps.fonttype'        : 42,
})

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILES   = 8
WINDOW_SIZE = 10
N_PANEL_ROWS = 4   # rows per GPU column
N_PANEL_COLS = 2   # panels per row within one GPU column
N_GPUS       = 3

GPU_NAMES  = ['A800', 'g3090Ti', 'g4090']
GPU_LABELS = ['(a) NVIDIA A800', '(b) NVIDIA RTX 3090 Ti', '(c) NVIDIA RTX 4090']

# Full ACM two-column text width
# Each GPU column ≈ 2.10 in wide, 2 panel cols → each panel ≈ 0.95 in wide.
# For ~square panels: panel height ≈ 0.95 in, 4 rows → ~3.8 in panel area.
# Add space for col headers, x-labels, shared legend → FIG_H ≈ 5.20 in.
FIG_W = 6.50
FIG_H = 5.20

COLORS = {
    'original': '#A8B8CC',
    'mean':     '#E69F00',
    'max':      '#009E73',
    'baseline': '#CC3333',
    'peak':     '#D55E00',
}
LS_MEAN = '-'
LS_MAX  = (0, (4, 1.8))
LS_BASE = (0, (3, 1.2, 1, 1.2))

DATA_ROOT = "./Test_RL_Res"
OUT_DIR   = "./outputs"


# ── Helper: load one GPU's files ───────────────────────────────────────────────
def load_files(gpu_name):
    folder  = os.path.join(DATA_ROOT, gpu_name, 'reward_list')
    pattern = os.path.join(folder, 'reward_list_*.pt')
    files   = sorted(glob.glob(pattern))[:MAX_FILES]
    if not files:
        print(f"[WARN] No files found: {pattern}")
    return files


# ── Helper: plot one panel ─────────────────────────────────────────────────────
def plot_panel(ax, file_path, is_leftmost_col, is_bottom_row, collect_legend):
    ax.set_facecolor('#F4F6F9')

    model_name = (os.path.basename(file_path)
                  .replace('reward_list_', '')
                  .replace('.pt', ''))

    data     = torch.load(file_path, map_location='cpu')
    data     = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)
    baseline = float(data[0])
    rewards  = data[1:]
    epochs   = np.arange(1, len(rewards) + 1)

    if len(rewards) >= WINDOW_SIZE:
        moving_avg    = np.convolve(rewards,
                                    np.ones(WINDOW_SIZE) / WINDOW_SIZE,
                                    mode='valid')
        moving_max    = np.array([np.max(rewards[j:j + WINDOW_SIZE])
                                   for j in range(len(rewards) - WINDOW_SIZE + 1)])
        moving_epochs = np.arange(WINDOW_SIZE, len(rewards) + 1)
    else:
        moving_avg = moving_max = moving_epochs = None

    line1, = ax.plot(epochs, rewards,
                     color=COLORS['original'], alpha=0.38, lw=0.5,
                     label='Raw data')

    line2 = line3 = None
    if moving_avg is not None:
        line2, = ax.plot(moving_epochs, moving_avg,
                         color=COLORS['mean'], lw=1.2, ls=LS_MEAN,
                         label=f'{WINDOW_SIZE}-ep moving avg')
        line3, = ax.plot(moving_epochs, moving_max,
                         color=COLORS['max'],  lw=1.0, ls=LS_MAX,
                         label=f'{WINDOW_SIZE}-ep moving max')

    line4 = ax.axhline(y=baseline,
                       color=COLORS['baseline'], ls=LS_BASE,
                       lw=0.8, alpha=0.88, label='Baseline')

    max_idx    = int(np.argmax(rewards))
    max_reward = float(rewards[max_idx])
    scatter    = ax.scatter(epochs[max_idx], max_reward,
                            color=COLORS['peak'], s=38, marker='*',
                            edgecolors='#660000', linewidths=0.4,
                            zorder=10, label='Global max')

    # annotation box — 2 decimal places
    ax.text(0.97, 0.03,
            f"Base:{baseline:.2f}\nMax:{max_reward:.2f}",
            transform=ax.transAxes,
            fontsize=5.0, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#444444', alpha=0.90, linewidth=0.5))

    for sp in ax.spines.values():
        sp.set_linewidth(0.55)
        sp.set_color('#222222')

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.grid(True, ls=':', lw=0.28, alpha=0.22, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(model_name, fontsize=6.5, fontweight='bold',
                 pad=2.5, color='#1a1a2e', fontfamily='serif')

    if is_leftmost_col:
        ax.set_ylabel('Gain (ms)', fontsize=6.0, labelpad=2)
        ax.tick_params(axis='y', labelleft=True, labelright=False)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False, labelright=False,
                       left=True, right=False)

    if is_bottom_row:
        ax.set_xlabel('Epoch', fontsize=6.0, labelpad=2)

    handles = labels = None
    if collect_legend and line2 is not None:
        handles = [line1, line2, line3, line4, scatter]
        labels  = [
            'Raw data',
            f'{WINDOW_SIZE}-ep moving avg',
            f'{WINDOW_SIZE}-ep moving max',
            'Baseline',
            'Global max',
        ]
    return handles, labels


# ── Build combined figure ──────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

# Overall figure title — sits at very top
fig.text(0.5, 0.997,
         'DQN-Based NN Compilation — Fusion Gain (ms) across RL Convergence Epochs',
         ha='center', va='top', fontsize=8.5,
         fontfamily='serif', fontweight='bold', color='#1a1a2e')

# Outer GridSpec: 1 row × 3 cols — one col per GPU
# Each cell will contain an inner 4×2 GridSpec
# Outer GridSpec: 1 row × 3 cols — one col per GPU
# Tight outer wspace; dividers live INSIDE each GPU block's inner wspace
outer = GridSpec(
    1, N_GPUS,
    figure=fig,
    left=0.07, right=0.99,
    top=0.895,
    bottom=0.13,
    wspace=0.35,   # gap between GPU blocks — divider lives here
)

from matplotlib.lines import Line2D

shared_handles = shared_labels = None

for gpu_idx, (gpu_name, gpu_label) in enumerate(zip(GPU_NAMES, GPU_LABELS)):
    files = load_files(gpu_name)
    if not files:
        continue

    # Inner GridSpec: wide wspace so the middle gap is used for the divider
    inner = outer[0, gpu_idx].subgridspec(
        N_PANEL_ROWS, N_PANEL_COLS,
        hspace=0.72,
        wspace=0.18,   # normal inner gap, no divider here
    )

    # ── Column header ──────────────────────────────────────────────────────────
    ss  = outer[0, gpu_idx].get_position(fig)
    x_c = (ss.x0 + ss.x1) / 2
    y_hd = outer.top + (1.0 - outer.top) * 0.30

    fig.text(x_c, y_hd, gpu_label,
             ha='center', va='center',
             fontsize=8.0, fontweight='bold',
             color='#1a1a2e', fontfamily='serif')

    # ── Draw all panels first, collect per-col bbox ────────────────────────────
    axes_grid = {}   # (r, c) -> ax
    panel_idx = 0
    for r in range(N_PANEL_ROWS):
        for c in range(N_PANEL_COLS):
            if panel_idx >= len(files):
                dummy = fig.add_subplot(inner[r, c])
                dummy.set_axis_off()
                axes_grid[(r, c)] = dummy
                panel_idx += 1
                continue

            ax = fig.add_subplot(inner[r, c])
            is_left   = (c == 0)
            is_bottom = (r == N_PANEL_ROWS - 1)
            collect   = (gpu_idx == 0 and panel_idx == 0)

            h, l = plot_panel(ax, files[panel_idx],
                              is_leftmost_col=is_left,
                              is_bottom_row=is_bottom,
                              collect_legend=collect)
            if h is not None:
                shared_handles, shared_labels = h, l

            axes_grid[(r, c)] = ax
            panel_idx += 1

    # ── Vertical divider BETWEEN GPU blocks (after block 0 and block 1) ──────
    # Draw after rendering panels so inner bbox coords are available.
    # Position = midpoint between right edge of this block's right panel col
    # and left edge of next block's left panel col.
    # We store each block's inner right-col bbox for use in next iteration.
    ss_inner_right = inner[0, 1].get_position(fig)   # rightmost panel col bbox
    ss_inner_left  = inner[0, 0].get_position(fig)   # leftmost panel col bbox

    if gpu_idx > 0:
        # x_prev_right was stored from previous GPU block
        x_div = x_prev_right + (ss_inner_left.x0 - x_prev_right) * 0.25
        div = Line2D([x_div, x_div],
                     [outer.bottom - 0.01, outer.top + 0.04],
                     transform=fig.transFigure,
                     color='#999999', lw=0.9,
                     linestyle=(0, (6, 3)),
                     clip_on=False)
        fig.add_artist(div)

    # Save right edge of this block's right panel for next iteration
    x_prev_right = ss_inner_right.x1

# ── Shared legend — centred below all columns, max 2 rows ─────────────────────
# 5 items × ncol=5 → 1 row; fall back to ncol=3 (2 rows) if too wide
if shared_handles:
    leg = fig.legend(
        shared_handles, shared_labels,
        loc='lower center',
        bbox_to_anchor=(0.525, 0.002),
        ncol=5,              # try 1 row; reduce to 3 if labels are long
        fontsize=6.5,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=1.0,
        borderpad=0.40,
        labelspacing=0.22,
        framealpha=0.97,
        edgecolor='#444444',
    )
    leg.get_frame().set_linewidth(0.5)

# ── Save ───────────────────────────────────────────────────────────────────────
base = os.path.join(OUT_DIR, 'fusion_gain_combined')
fig.savefig(base + '.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(base + '.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {base}.pdf / .png")

print(r"""
LaTeX usage:
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{fusion_gain_combined.pdf}
  \caption{Fusion latency improvement (ms) across RL convergence epochs on
    (a)~NVIDIA A800, (b)~NVIDIA RTX~3090~Ti, and (c)~NVIDIA RTX~4090
    (batch size = 128).
    In each sub-figure, the eight panels correspond to the eight evaluated
    neural network models: Swin Transformer (Swin-T), EdgeNeXt, StarNet,
    ShuffleNet, MobileViT, NAFNet, BERT, and NeRF.
    \emph{Raw data}: per-epoch reward; \emph{moving avg/max}: 10-epoch
    sliding-window average and maximum; \emph{Baseline}: initial reward at
    epoch~1; $\bigstar$: global maximum reward.}
  \label{fig:reward_all}
\end{figure*}
""")