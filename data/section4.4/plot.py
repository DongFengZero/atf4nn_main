"""
plot_dqn_acm.py  —  ACM double-column (full-width) format
==========================================================
DQN-Based Neural Network Compilation — Per-Epoch Reward (ms) and Iteration Time (s)

ACM double-column target:
  Layout  : 2 rows × 4 cols = 8 panels
  Width   : 7.16 in  (= ACM double-column / full text-block width)
  Height  : 3.80 in  (compact two-row layout)
  Fonts   : Times New Roman + STIX math, 6–8 pt
  Output  : PDF (Type-42 embedded, 600 dpi) + PNG
"""

import zipfile, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')

matplotlib.rcParams.update({
    # ACM-compatible fonts
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size'          : 8.5,
    'mathtext.fontset'   : 'stix',
    'text.usetex'        : False,
    # Axes
    'axes.linewidth'     : 0.6,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.unicode_minus' : False,
    # Ticks
    'xtick.direction'    : 'out', 'ytick.direction'    : 'out',
    'xtick.major.size'   : 2.5,   'ytick.major.size'   : 2.5,
    'xtick.major.width'  : 0.55,  'ytick.major.width'  : 0.55,
    'xtick.labelsize'    : 6.5,   'ytick.labelsize'    : 6.5,
    # Legend
    'legend.framealpha'  : 0.96,
    'legend.edgecolor'   : '#d0d0d0',
    'legend.fontsize'    : 6.5,
    'legend.handlelength': 1.4,
    'legend.borderpad'   : 0.35,
    'legend.labelspacing': 0.20,
    # Output
    'figure.dpi'         : 150,
    'savefig.dpi'        : 600,
    'pdf.fonttype'       : 42,   # Type-42 embed — ACM requirement
    'ps.fonttype'        : 42,
})

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects

# ── Model metadata ─────────────────────────────────────────────────────────────
MODELS = ['swin_transformer', 'edgenext', 'StarNet', 'shufflenet',
          'mobilevit', 'NAFNet', 'bert', 'NeRF']
LABELS = ['Swin Transformer', 'EdgeNeXt', 'StarNet', 'ShuffleNet',
          'MobileViT', 'NAFNet', 'BERT', 'NeRF']
COLORS = ['#0077BB', '#EE7733', '#009988', '#CC3311',
          '#33BBEE', '#AA3377', '#555555', '#EE3377']

YLIM_T = {
    'swin_transformer': (0, 110), 'edgenext':  (0, 22),
    'StarNet':          (0, 11),  'shufflenet': (0, 9),
    'mobilevit':        (0, 20),  'NAFNet':     (0, 16),
    'bert':             (0, 20),  'NeRF':       (0, 0.18),
}
TIME_COLOR = '#5A7FA8'

# Greyscale-safe line styles for time curve
LS_TIME = (0, (4, 1.8))   # dashed

# ── Helpers ────────────────────────────────────────────────────────────────────
def rolling_max(arr, w=10):
    out = []
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out.append(max(arr[s:i+1]))
    return np.array(out, dtype=float)

def smooth_t(arr, w=50):
    out = []
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out.append(sum(arr[s:i+1]) / (i - s + 1))
    return np.array(out, dtype=float)

DS = 10   # down-sample stride

# ── Figure layout ──────────────────────────────────────────────────────────────
# ACM double-column (full page width) = 7.16 in.
# 8 panels arranged in 2 rows × 4 cols.
FIG_W = 7.16   # ← changed from 3.50 to full ACM double-column width
FIG_H = 3.80   # ← changed from 6.30 to compact two-row height

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='#FAFAFA')
gs  = GridSpec(2, 4, figure=fig,          # ← changed from (4, 2) to (2, 4)
               hspace=0.62, wspace=0.58,
               left=0.06, right=0.97, top=0.890, bottom=0.18)
               # ↑ left/right margins tightened for wider canvas;
               #   top reduced to leave room for title; bottom for legend

fig.text(0.5, 0.999,
         'DQN-Based Neural Network Compilation\n'
         'Per-Epoch Reward (ms) and Iteration Time (s)',
         ha='center', va='top', fontsize=8.0,
         fontfamily='serif', fontweight='bold', color='#1a1a2e',
         linespacing=1.35)

# ── Per-panel loop ─────────────────────────────────────────────────────────────
for idx, (m, lb, c) in enumerate(zip(MODELS, LABELS, COLORS)):
    row, col = divmod(idx, 4)             # ← changed divisor from 2 to 4
    ax1 = fig.add_subplot(gs[row, col])
    ax1.set_facecolor('#F4F6F9' if (row + col) % 2 == 0 else '#FFFFFF')
    ax1.set_box_aspect(1)   # enforce square panel
    ax2 = ax1.twinx()

    # ── Load data ──────────────────────────────────────────────────────────
    with zipfile.ZipFile(f'./res_time/{m}/reward_list.pt') as z:
        with z.open('reward_list/data.pkl') as f:
            r = np.array(list(pickle.load(f)), dtype=float)
    with zipfile.ZipFile(f'./res_time/{m}/time_list.pt') as z:
        with z.open('time_list/data.pkl') as f:
            t = np.array(list(pickle.load(f)), dtype=float)

    N          = len(r)
    rmax_full  = rolling_max(r, w=10)
    st_full    = smooth_t(t, w=50)
    baseline_r = float(r[0])
    best_ep_0  = int(np.argmax(rmax_full))
    best_ep    = best_ep_0 + 1
    best_r     = float(rmax_full[best_ep_0])

    idx_ds    = np.arange(0, N, DS)
    ep_plot   = idx_ds + 1
    r_plot    = r[idx_ds]
    rmax_plot = rmax_full[idx_ds]
    st_plot   = st_full[idx_ds]

    # ── y-axis range ───────────────────────────────────────────────────────
    r_body = r[1:]
    lo     = float(np.percentile(r_body, 5))
    hi     = float(r_body.max())
    span   = hi - lo
    pad    = span * 0.55
    ylim_r = (lo - pad, hi + pad * 0.6)

    # ── Raw reward ─────────────────────────────────────────────────────────
    ax1.plot(ep_plot, r_plot, color=c, lw=0.6, alpha=0.28, zorder=3,
             label='Reward / ep (ms)')

    # ── Rolling max ────────────────────────────────────────────────────────
    ax1.plot(ep_plot, rmax_plot, color=c, lw=1.5, alpha=0.93, zorder=4,
             label='10-ep rolling max')
    ax1.fill_between(ep_plot, ylim_r[0], rmax_plot,
                     alpha=0.09, color=c, zorder=2, lw=0)

    # ── ep1 baseline dashed ────────────────────────────────────────────────
    ax1.axhline(baseline_r, color=c, lw=0.8, ls=(0, (5, 4)),
                alpha=0.55, zorder=3,
                label=f'ep1 baseline  {baseline_r:.3f} ms')

    ax1.set_xlim(1, N)
    ax1.set_ylim(*ylim_r)

    # ── Best-epoch star ────────────────────────────────────────────────────
    ax1.scatter([best_ep], [best_r], s=80, marker='*',
                color='#CC0000', zorder=9, edgecolors='white', linewidths=0.5,
                label=f'Best @ ep {best_ep}')

    x_off = 280 if best_ep < 3800 else -780
    ha    = 'left' if x_off > 0 else 'right'
    y_ann = min(best_r + span * 0.20, ylim_r[1] - span * 0.08)
    txt   = ax1.annotate(
        f'ep {best_ep}',
        xy=(best_ep, best_r),
        xytext=(best_ep + x_off, y_ann),
        fontsize=6.5, color='#CC0000', ha=ha,
        arrowprops=dict(arrowstyle='->', color='#CC0000',
                        lw=0.5, alpha=0.85,
                        connectionstyle='arc3,rad=0.15'))
    txt.set_path_effects(
        [patheffects.withStroke(linewidth=1.5, foreground='white')])

    # ── Iter time ──────────────────────────────────────────────────────────
    st_clip = np.clip(st_plot, 0, YLIM_T[m][1])
    ax2.fill_between(ep_plot, 0, st_clip,
                     alpha=0.13, color=TIME_COLOR, zorder=1, lw=0)
    ax2.plot(ep_plot, st_clip,
             color=TIME_COLOR, lw=1.1, ls=LS_TIME, alpha=0.75, zorder=3,
             label='Iter Time (s)')
    ax2.set_xlim(1, N)
    ax2.set_ylim(*YLIM_T[m])

    # ── Right spine (time axis) ────────────────────────────────────────────
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(0.6)
    ax2.spines['right'].set_color(TIME_COLOR)
    ax2.tick_params(axis='y', colors=TIME_COLOR, labelsize=6.5, pad=1.5)
    ax2.yaxis.label.set_color(TIME_COLOR)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

    # ── Left spine (reward axis) ───────────────────────────────────────────
    ax1.spines['left'].set_color(c)
    ax1.spines['left'].set_linewidth(0.7)
    ax1.spines['bottom'].set_linewidth(0.55)
    ax1.spines['bottom'].set_color('#888888')
    ax1.tick_params(axis='y', colors=c, labelsize=6.5, pad=1.5)
    ax1.tick_params(axis='x', colors='#555555', labelsize=6.5)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

    ax1.set_xticks([1, 1000, 2000, 3000, 4000, 5000])
    ax1.set_xticklabels(['1', '1k', '2k', '3k', '4k', '5k'],
                        fontsize=6.5, color='#555555')

    # ── Panel title ────────────────────────────────────────────────────────
    ax1.set_title(lb, fontsize=8.5, fontweight='bold',
                  pad=3, color=c, fontfamily='serif')

    # ── Axis labels ────────────────────────────────────────────────────────
    # Show y-axis labels only on leftmost column (col == 0) to save space
    if col == 0:
        ax1.set_ylabel('Reward (ms)', fontsize=7.5, color=c, labelpad=2)
    else:
        ax1.set_ylabel('')
        ax1.tick_params(axis='y', labelleft=True)   # still show tick values

    # Show right-axis label only on rightmost column (col == 3) to save space
    if col == 3:
        ax2.set_ylabel('Time (s)', fontsize=7.5, color=TIME_COLOR, labelpad=2)
    else:
        ax2.set_ylabel('')

    ax1.set_xlabel('Epoch', fontsize=7.0, color='#444444', labelpad=1.5)

    ax1.grid(True, axis='both', ls=':', lw=0.35, alpha=0.22, zorder=0)

    # collect handles from the last panel for the shared legend
    if idx == len(MODELS) - 1:
        _h1, _l1 = ax1.get_legend_handles_labels()
        _h2, _l2 = ax2.get_legend_handles_labels()
        shared_handles = _h1 + _h2
        shared_labels  = _l1 + _l2

# ── Shared legend — centred below the last row ────────────────────────────────
# Normalise variable-content labels to generic text
shared_labels = [
    'Best epoch'   if l.startswith('Best @')      else l for l in shared_labels]
shared_labels = [
    'ep1 baseline' if l.startswith('ep1 baseline') else l for l in shared_labels]

leg = fig.legend(
    shared_handles, shared_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.00),
    ncol=5,                      # ← 5 cols fits nicely in wider figure
    fontsize=7.0,
    handlelength=1.5,
    handletextpad=0.35,
    columnspacing=0.8,
    borderpad=0.45,
    labelspacing=0.22,
    framealpha=0.96,
    edgecolor='#d0d0d0',
)
leg.get_frame().set_linewidth(0.4)

gs.update(bottom=0.18)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig('./outputs/fig_reward_time_acm.png',
            dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
fig.savefig('./outputs/fig_reward_time_acm.pdf',
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Done.")