# app/analyses/discourse.py
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts, JsCode
from scipy.stats import entropy as scipy_entropy

from app.data import DIS, REAL, SPEAK, SENT, EMOT, COLS_4WAY

_REAL_ABB  = {'real_sickness': 'rs', 'not_real_sickness': 'nrs'}
_SPEAK_ABB = {'writer': 'wr', 'third_voice': 'tv', 'disease': 'di',
              'not_conclusive': 'nc', 'family': 'fa', 'celebrity': 'ce',
              'friends_colleagues': 'fr'}
_EMOT_ABB  = {'other': 'o', 'joy': 'j', 'anger': 'a', 'disgust': 'd',
              'fear': 'f', 'sadness': 'sa', 'surprise': 'su'}

def _abbrev(row) -> str:
    return '/'.join([
        _REAL_ABB.get(row[REAL], row[REAL][:3]),
        _SPEAK_ABB.get(row[SPEAK], row[SPEAK][:2]),
        row[SENT][:3],
        _EMOT_ABB.get(row[EMOT], row[EMOT][:2]),
    ])

def compute_discourse(df):
    """Return (entropy_dict, top5_dict) for the discourse heatmap."""
    dis_labels = sorted(df[DIS].unique())
    entropy_d, top5_d = {}, {}
    for dis in dis_labels:
        sub = df[df[DIS] == dis]
        n = len(sub)
        combo = (sub.groupby(COLS_4WAY).size()
                    .reset_index(name='count')
                    .sort_values('count', ascending=False)
                    .reset_index(drop=True))
        probs = combo['count'] / combo['count'].sum()
        entropy_d[dis] = float(scipy_entropy(probs, base=2))
        top = combo.head(5).copy()
        top['pct']   = top['count'] / n * 100
        top['label'] = top.apply(_abbrev, axis=1)
        top5_d[dis]  = top[['label', 'pct']].to_dict('records')
    return entropy_d, top5_d

def render(df: pd.DataFrame) -> None:
    st.subheader('Dominant Discourse Mode Per Disease')
    entropy_d, top5_d = compute_discourse(df)
    dis_sorted = sorted(entropy_d, key=entropy_d.get)

    # ── ECharts: entropy bar ──────────────────────────────────────────────────
    ent_options = {
        "title": {"text": "Discourse Entropy per Disease", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "8%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value", "name": "Shannon entropy (bits)"},
        "yAxis": {"type": "category", "data": [d.capitalize() for d in dis_sorted]},
        "series": [{
            "type": "bar",
            "data": [round(entropy_d[d], 3) for d in dis_sorted],
            "itemStyle": {"color": "#5470c6"},
            "label": {"show": True, "position": "right", "formatter": "{c}"},
        }],
    }
    st_echarts(options=ent_options, height="400px")

    # ── ECharts: top-5 heatmap ────────────────────────────────────────────────
    top_k = 5
    rank_labels = [f"Rank {k+1}" for k in range(top_k)]
    pct_data = []
    lbl_grid = [['' for _ in range(top_k)] for _ in range(len(dis_sorted))]
    for i, dis in enumerate(dis_sorted):
        for j, entry in enumerate(top5_d[dis][:top_k]):
            pct_data.append([j, i, round(entry['pct'], 1)])
            lbl_grid[i][j] = entry['label']

    hm_options = {
        "title": {"text": "Top-5 Discourse Archetypes per Disease", "left": "center"},
        "tooltip": {
            "formatter": JsCode(
                f"function(p){{var g={lbl_grid};return g[p.data[1]][p.data[0]]+'<br/>'+p.data[2]+'%';}}"
            )
        },
        "grid": {"left": "15%", "right": "5%", "bottom": "15%", "top": "12%"},
        "xAxis": {"type": "category", "data": rank_labels, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": [d.capitalize() for d in dis_sorted], "splitArea": {"show": True}},
        "visualMap": {
            "min": 0,
            "max": max(e['pct'] for entries in top5_d.values() for e in entries),
            "calculable": True,
            "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#ffffcc", "#fd8d3c", "#800026"]},
        },
        "series": [{
            "type": "heatmap",
            "data": pct_data,
            "label": {
                "show": True,
                "formatter": JsCode(
                    f"function(p){{var g={lbl_grid};return g[p.data[1]][p.data[0]]+'\\n'+p.data[2]+'%';}}"
                ),
            },
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }],
    }
    st_echarts(options=hm_options, height="500px")

    # ── Download matplotlib version ───────────────────────────────────────────
    # Rebuild the original matplotlib figure for download
    n_dis2, top_k2 = len(dis_sorted), 5
    pct_mat2 = np.full((n_dis2, top_k2), np.nan)
    lbl_mat2 = np.full((n_dis2, top_k2), '', dtype=object)
    ent_col = np.array([entropy_d[d] for d in dis_sorted])
    for i, dis in enumerate(dis_sorted):
        for j, entry in enumerate(top5_d[dis][:top_k2]):
            pct_mat2[i, j] = entry['pct']
            lbl_mat2[i, j] = entry['label']

    fig_h = max(10, n_dis2 * 0.75)
    fig_mpl = plt.figure(figsize=(16, fig_h))
    gs = gridspec.GridSpec(1, 2, figure=fig_mpl, width_ratios=[0.55, top_k2], wspace=0.04)
    ax_ent = fig_mpl.add_subplot(gs[0, 0])
    ax_hm  = fig_mpl.add_subplot(gs[0, 1])

    norm_ent = mcolors.Normalize(vmin=max(0, ent_col.min() - 0.1), vmax=ent_col.max() + 0.1)
    ax_ent.imshow(ent_col.reshape(-1, 1), aspect='auto', cmap='viridis', norm=norm_ent)
    font_s = max(6.5, 9.0 - n_dis2 * 0.18)
    for i, val in enumerate(ent_col):
        nv = norm_ent(val)
        ax_ent.text(0, i, f'{val:.2f}', ha='center', va='center', fontsize=font_s,
                    color='white' if nv < 0.55 else '#111', fontweight='bold')
    ax_ent.set_xticks([0]); ax_ent.set_xticklabels(['H'], fontsize=8.5)
    ax_ent.xaxis.set_label_position('top'); ax_ent.xaxis.tick_top()
    ax_ent.set_yticks(range(n_dis2))
    ax_ent.set_yticklabels([d.capitalize() for d in dis_sorted], fontsize=9.5)
    ax_ent.tick_params(length=0)
    for y in np.arange(-0.5, n_dis2, 1): ax_ent.axhline(y, color='white', lw=1.2)
    ax_ent.set_xlim(-0.5, 0.5)

    vmax2 = np.nanmax(pct_mat2) * 1.05
    ax_hm.imshow(np.where(np.isnan(pct_mat2), 0, pct_mat2), aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax2)
    for i in range(n_dis2):
        for j in range(top_k2):
            if not np.isnan(pct_mat2[i, j]):
                nv = pct_mat2[i, j] / vmax2
                ax_hm.text(j, i, f'{lbl_mat2[i,j]}\n{pct_mat2[i,j]:.1f}%',
                           ha='center', va='center', fontsize=font_s,
                           color='white' if nv > 0.55 else '#222', linespacing=1.35)
    ax_hm.set_xticks(range(top_k2))
    ax_hm.set_xticklabels([f'Rank {k+1}' for k in range(top_k2)], fontsize=9)
    ax_hm.set_yticks(range(n_dis2)); ax_hm.set_yticklabels([])
    ax_hm.tick_params(length=0)
    for x in np.arange(-0.5, top_k2, 1): ax_hm.axvline(x, color='white', lw=1.2)
    for y in np.arange(-0.5, n_dis2, 1): ax_hm.axhline(y, color='white', lw=1.2)
    fig_mpl.suptitle('Dominant 4-Way Discourse Archetypes per Disease', fontsize=10.5, fontweight='bold', y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='discourse.png', mime='image/png')
