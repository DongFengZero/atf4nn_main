"""
plot_dram.py  —  ACM single-column half-width edition
======================================================
Parse Nsight Compute CSVs from run_profile.sh and plot
flash_attn DRAM traffic vs Tile S (S=768..2048, step 128).

ACM single-column half-width target:
  Figure width  : 3.33 in  (= ACM single-column text block width)
  Figure height : 4.40 in  (taller to accommodate all original annotations)
  Font sizes    : 5.5–7 pt
  Output        : PDF (Type-42 embedded) + PNG 600 dpi

ALL original content preserved:
  - Both % label rows (C1->C2 purple, C2->Meas red) at every point
  - Colour-coded explanation text box
  - Right-side endpoint value labels
  - Shaded bands, all three curves, scatter markers
  - Full title and axis labels

CSV ID encoding (100 reps x 3 kernels = IDs 0..299):
    rep r, kernel k  ->  ID = r*3 + k
    k=0  tiled_gemm_h   K0: Q x K^T -> S'
    k=1  row_softmax_h  K1: softmax in-place
    k=2  tiled_gemm_h   K2: S' x V  -> O

Theory curves (S' > L2 regime):
    Curve 1 = (Q+K) + 3S' + (S'+V) + (S'-L2)
    Curve 2 = Curve 1 + (S'-L2)

Measured: min of 100 reps (dram_read + dram_write).

Usage:
    python3 plot_dram_acm.py [csv_dir] [output_png] [output_tex]
"""

import glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ── ACM rcParams ───────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size'          : 7,
    'mathtext.fontset'   : 'stix',
    'text.usetex'        : False,
    'axes.linewidth'     : 0.6,
    'axes.labelsize'     : 7,
    'axes.titlesize'     : 7,
    'axes.titlepad'      : 4,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.unicode_minus' : False,
    'xtick.direction'    : 'out',
    'ytick.direction'    : 'out',
    'xtick.major.size'   : 3,   'ytick.major.size'  : 3,
    'xtick.minor.size'   : 1.5, 'ytick.minor.size'  : 1.5,
    'xtick.major.width'  : 0.6, 'ytick.major.width' : 0.6,
    'xtick.labelsize'    : 6,   'ytick.labelsize'   : 6,
    'legend.fontsize'    : 6,
    'legend.framealpha'  : 1.0,
    'legend.edgecolor'   : '#BBBBBB',
    'legend.handlelength': 2.2,
    'legend.handletextpad': 0.4,
    'legend.borderpad'   : 0.4,
    'legend.labelspacing': 0.3,
    'lines.linewidth'    : 1.2,
    'lines.markersize'   : 3.5,
    'figure.dpi'         : 300,
    'savefig.dpi'        : 600,
    'pdf.fonttype'       : 42,   # Type-42 embed — ACM requirement
    'ps.fonttype'        : 42,
})

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants ─────────────────────────────────────────────────────────────────
L2_BYTES    = 6 * 1024 * 1024
D           = 64
T           = 16
N_KERNELS   = 3
METRICS     = ('dram_read_bytes', 'dram_write_bytes')
METRICS_NEW = ('dram__bytes_read.sum', 'dram__bytes_write.sum')

# Original colours preserved
C1c      = '#2e7d32'
C2c      = '#6a1b9a'
CMc      = '#c62828'
FILL_12  = '#e8f5e9'
FILL_2M  = '#ede7f6'

# Greyscale-safe line styles added for B&W print legibility
LS_C1  = '-'
LS_C2  = (0, (5, 2))
LS_CM  = (0, (3, 1.5, 1, 1.5))
MK_C1, MK_C2, MK_CM = 'o', 's', '^'

# ACM half-column figure size
FIG_W = 3.33
FIG_H = 3.33


# ── Geometry / Theory ─────────────────────────────────────────────────────────
def tile_geometry(S):
    Br = max(S * 128 // (D * 2), T)
    Br = (Br // T) * T
    L  = Br * 8
    return dict(Br=Br, L=L, Q=Br*D*2, K=L*D*2, V=L*D*2, Sp=Br*L*2)

def theory(S):
    g     = tile_geometry(S)
    Sp    = g['Sp']
    extra = Sp - L2_BYTES
    c1    = g['Q'] + g['K'] + 3*Sp + (Sp + g['V']) + extra
    c2    = c1 + extra
    return c1 / 1024 / 1024, c2 / 1024 / 1024


# ── CSV parsing ───────────────────────────────────────────────────────────────
def load_csv(path: str, n_reps: int = 100) -> np.ndarray:
    df = pd.read_csv(path, skiprows=2)
    df['Metric Value'] = (
        df['Metric Value'].astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    all_metrics = set(df['Metric Name'].unique())
    metrics = METRICS_NEW if METRICS_NEW[0] in all_metrics else METRICS
    rw = np.zeros(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        total = 0.0
        for kid in range(N_KERNELS):
            gid = rep * N_KERNELS + kid
            sub = df[df['ID'] == gid]
            for m in metrics:
                v = sub.loc[sub['Metric Name'] == m, 'Metric Value']
                if len(v):
                    total += float(v.iloc[0])
        rw[rep] = total
    return rw

def load_all(csv_dir: str, tiles: list = None, n_reps: int = 100) -> pd.DataFrame:
    if tiles is None:
        tiles = [768, 896, 1024, 1152, 1280,
                 1408, 1536, 1664, 1792, 1920, 2048]
    found = {}
    for f in glob.glob(os.path.join(csv_dir, 'flash_attn_S*.csv')):
        m = re.search(r'S(\d+)', os.path.basename(f))
        if m:
            found[int(m.group(1))] = f
    if not found:
        raise FileNotFoundError(f'No flash_attn_S*.csv in: {csv_dir}')
    rows = []
    for S in sorted(set(tiles) & set(found)):
        geo    = tile_geometry(S)
        c1, c2 = theory(S)
        rw     = load_csv(found[S], n_reps=n_reps)
        valid  = rw[rw > 0] / 1024 / 1024
        rows.append(dict(
            S            = S,
            Br           = geo['Br'],
            L            = geo['L'],
            Sp_MB        = geo['Sp'] / 1024 / 1024,
            C1_MB        = c1,
            C2_MB        = c2,
            meas_min_MB  = valid.min()       if len(valid) else np.nan,
            meas_mean_MB = valid.mean()      if len(valid) else np.nan,
            meas_std_MB  = valid.std(ddof=1) if len(valid) > 1 else np.nan,
            ok_reps      = int((rw > 0).sum()),
        ))
    return pd.DataFrame(rows).sort_values('S').reset_index(drop=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(df: pd.DataFrame, output_path: str = 'dram_vs_tile.png'):
    tile_S = df['S'].values
    r1     = df['C1_MB'].values
    r2     = df['C2_MB'].values
    m_min  = df['meas_min_MB'].values
    n      = len(tile_S)

    # ── y-axis range (same logic as original) ─────────────────────────────────
    ybot  = max(0.0, np.nanmin(r1) * 0.85)
    ytop  = np.nanmax(m_min) * 1.08
    yspan = ytop - ybot

    # Two rows of % labels above m_min — same spacing ratios as original
    dy_base = yspan * 0.055
    dy_step = yspan * 0.075
    label_headroom = dy_base + dy_step * 1.4 + yspan * 0.10
    ymax = ytop + label_headroom

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor='white')
    ax.set_facecolor('white')

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.yaxis.grid(True, color='#e0e0e0', lw=0.5, zorder=0)
    ax.xaxis.grid(True, color='#ebebeb', lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # ── Shaded bands ──────────────────────────────────────────────────────────
    ax.fill_between(tile_S, r1, r2,    color=FILL_12, alpha=0.90, zorder=1)
    ax.fill_between(tile_S, r2, m_min, color=FILL_2M, alpha=0.78, zorder=1)

    # ── Three main curves ─────────────────────────────────────────────────────
    ax.plot(tile_S, r1,
            color=C1c, lw=1.3, ls=LS_C1,
            marker=MK_C1, ms=3.5, mfc='white', mew=1.0, zorder=4,
            label=r'Curve 1: $X$-Partition')

    ax.plot(tile_S, r2,
            color=C2c, lw=1.3, ls=LS_C2,
            marker=MK_C2, ms=3.5, mfc='white', mew=1.0, zorder=4,
            label=r'Curve 2: $(X_1,X_2)$-Partition')

    ax.plot(tile_S, m_min,
            color=CMc, lw=1.2, ls=LS_CM,
            marker=MK_CM, ms=3.5, mfc='white', mew=1.0, zorder=5,
            label=r'Measured: dram_read $+$ dram_write  (min of 100 reps)')

    # ── Per-point % labels — ALL points, two rows (identical to original) ─────
    for i in range(n):
        s   = tile_S[i]
        v1, v2, vm = r1[i], r2[i], m_min[i]
        pct12 = int(round((v2 / v1 - 1) * 100))   # C1→C2
        pct2m = int(round((vm / v2 - 1) * 100))    # C2→Meas

        if i % 2 == 0:
            y_green = vm + dy_base
            y_red   = vm + dy_base + dy_step
        else:
            y_green = vm + dy_base + dy_step * 0.4
            y_red   = vm + dy_base + dy_step * 1.4

        ax.text(s, y_green, f'+{pct12}%',
                color=C2c, fontsize=5.5, ha='center', va='bottom',
                fontweight='bold')
        ax.text(s, y_red,   f'+{pct2m}%',
                color=CMc, fontsize=5.5, ha='center', va='bottom',
                fontweight='bold')

    # ── Colour-coded explanation box (same as original) ───────────────────────
    ax.text(0.99, 0.03,
            'purple +x% = C1\u2192C2     red +x% = C2\u2192Meas Min',
            transform=ax.transAxes,
            fontsize=5, ha='right', va='bottom', color='#444',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#ccc', alpha=0.85))

    # ── Right-side endpoint value labels (collision-aware, same as original) ──
    x_end   = float(tile_S[-1])
    x_range = float(tile_S[-1] - tile_S[0])
    items   = sorted([(r1[-1], C1c), (r2[-1], C2c), (m_min[-1], CMc)],
                     key=lambda x: x[0])
    min_gap = yspan * 0.038
    placed  = []
    for val, col in items:
        y = val
        for py in placed:
            if abs(y - py) < min_gap:
                y = py + min_gap
        placed.append(y)
        ax.text(x_end + x_range * 0.015, y, f'{val:.1f}',
                color=col, fontsize=6, va='center', ha='left',
                fontweight='bold', clip_on=False)

    # ── Axes limits & ticks ───────────────────────────────────────────────────
    ax.set_xlim(tile_S[0] - x_range * 0.04, x_end + x_range * 0.09)
    ax.set_ylim(ybot, ymax)

    # All x-ticks shown, rotated 45° to fit narrow column
    ax.set_xticks(tile_S)
    ax.set_xticklabels([str(s) for s in tile_S], rotation=45, ha='right',
                       fontsize=5.5)

    raw_step = (ytop - ybot) / 6
    step = max(25, int(round(raw_step / 25)) * 25)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(step / 2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # ── Title (same text as original, line-wrapped for narrow width) ──────────
    ax.set_xlabel('Tile Size', fontsize=7, labelpad=3)
    ax.set_ylabel('DRAM Traffic  (MB)', fontsize=7, labelpad=3)
    ax.set_title(
        r'Flash_Attention  DRAM Traffic vs Tilesize'           '\n'
        r'($S^\prime\!>\!S$,  Tilesize $=768\!\ldots\!2048$,  step 128)'  '\n'
        r'RTX 3090 Ti $\cdot$ $D\!=\!64$ $\cdot$ L2 $S\!=\!6\,\mathrm{MB}$'  '\n'
        r'100 reps $\cdot$ Measured $=$ min of 100 reps',
        fontsize=6, fontweight='bold', pad=4, linespacing=1.45,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(loc='upper left', fontsize=5.5, frameon=True,
              facecolor='white', edgecolor='#BBBBBB',
              borderpad=0.4, labelspacing=0.3,
              handlelength=2.2, handletextpad=0.4)

    # ── Save PDF + PNG ────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.5)
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f'Saved: {pdf_path}')
    plt.close(fig)
    return fig, ax


# ── LaTeX table (unchanged) ───────────────────────────────────────────────────
def gen_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r'% flash\_attn DRAM traffic: theory vs measured',
        r'% Auto-generated by plot_dram.py',
        r'%',
        r'% Required packages: booktabs',
        r'% Add to preamble:',
        r'%   \usepackage{booktabs}',
        r'',
        r'\begin{table}[!ht]',
        r'\centering',
        r'\caption{%',
        r'  flash\_attn DRAM traffic: theory vs.\ measured'
        r'  ($S^\prime > L_2$, $S = 768\ldots2048$, step 128).\\[2pt]',
        r'  \small',
        r'  $\mathbf{C1}=(Q+K)+3S^\prime+(S^\prime+V)+(S^\prime-L_2)$;\quad'
        r'  $\mathbf{C2}=\text{C1}+(S^\prime-L_2)$.\quad'
        r'  Ratios: $\text{Theory}/\text{Measured}\times100\%$.}',
        r'\label{tab:dram_traffic}',
        r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{1.3}',
        r'\small',
        r'\begin{tabular}{r r r r r r r r r}',
        r'\toprule',
        r'$S$ & $S^\prime$ (MB) & $B_r$ & $L$'
        r'  & \textbf{C1} (MB)'
        r'  & \textbf{C2} (MB)'
        r'  & \textbf{Meas.\ min} (MB)'
        r'  & C1/Meas (\%)'
        r'  & C2/Meas (\%) \\',
        r'\midrule',
    ]
    for _, row in df.iterrows():
        S    = int(row['S'])
        Sp   = row['Sp_MB']
        Br   = int(row['Br'])
        L    = int(row['L'])
        c1   = row['C1_MB']
        c2   = row['C2_MB']
        mmin = row['meas_min_MB']
        rat1 = c1 / mmin * 100
        rat2 = c2 / mmin * 100
        lines.append(
            f'  {S} & {Sp:.3f} & {Br} & {L}'
            f' & {c1:.2f} & {c2:.2f}'
            f' & {mmin:.2f}'
            f' & {rat1:.1f} & {rat2:.1f} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    csv_dir    = sys.argv[1] if len(sys.argv) > 1 \
                 else './results/NVIDIA_GeForce_RTX_3090_Ti'
    output_png = sys.argv[2] if len(sys.argv) > 2 else 'dram_vs_tile.png'
    output_tex = sys.argv[3] if len(sys.argv) > 3 \
                 else output_png.replace('.png', '_table.tex')

    print(f'Loading CSVs from: {csv_dir}')
    df = load_all(csv_dir, n_reps=100)

    pd.set_option('display.float_format', '{:.3f}'.format)
    print('\nSummary:')
    print(df[['S', 'Sp_MB', 'C1_MB', 'C2_MB',
              'meas_min_MB', 'meas_mean_MB', 'meas_std_MB', 'ok_reps']]
          .to_string(index=False))

    plot(df, output_path=output_png)

    tex = gen_latex_table(df)
    with open(output_tex, 'w') as f:
        f.write(tex + '\n')
    print(f'LaTeX table : {output_tex}')
    print()
    print('─' * 62)
    print(tex)
    print('─' * 62)