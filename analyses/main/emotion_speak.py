# app/analyses/main/emotion_speak.py
import io
import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts, JsCode

from app.data import DIS, SPEAK, EMOT, SPEAK_LABELS

CORE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

EMOTION_COLORS = {
    'anger':    '#d62728',
    'disgust':  '#8c564b',
    'fear':     '#ff7f0e',
    'joy':      '#2ca02c',
    'sadness':  '#1f77b4',
    'surprise': '#9467bd',
    'other':    '#7f7f7f',
}


def compute_emotion_real(df: pd.DataFrame) -> pd.DataFrame:
    """For each (disease, speakertype) cell, return the dominant core emotion."""
    dis_labels = sorted(df[DIS].unique())
    spk_labels = sorted(df[SPEAK].unique())
    rows = []
    for dis in dis_labels:
        for spk in spk_labels:
            subset = df[(df[DIS] == dis) & (df[SPEAK] == spk)]
            n_total = len(subset)
            core_subset = subset[subset[EMOT].isin(CORE_EMOTIONS)]
            n_core = len(core_subset)
            if n_core == 0:
                other_subset = subset[subset[EMOT] == 'other']
                n_other = len(other_subset)
                if n_other > 0:
                    rows.append({'disease': dis, 'speakertype': spk,
                                 'dominant_emotion': 'other', 'dominant_pct': 1.0,
                                 'n_total': n_total, 'n_core': n_other})
                else:
                    rows.append({'disease': dis, 'speakertype': spk,
                                 'dominant_emotion': None, 'dominant_pct': np.nan,
                                 'n_total': n_total, 'n_core': 0})
            else:
                counts = core_subset[EMOT].value_counts()
                dominant = counts.index[0]
                rows.append({'disease': dis, 'speakertype': spk,
                             'dominant_emotion': dominant,
                             'dominant_pct': counts.iloc[0] / n_total,
                             'n_total': n_total, 'n_core': n_core})
    return pd.DataFrame(rows)


def _build_figure(emotion_df: pd.DataFrame) -> plt.Figure:
    emotion_pivot = emotion_df.pivot(index='disease', columns='speakertype', values='dominant_emotion')
    pct_pivot     = emotion_df.pivot(index='disease', columns='speakertype', values='dominant_pct')
    n_core_pivot  = emotion_df.pivot(index='disease', columns='speakertype', values='n_core')
    n_total_pivot = emotion_df.pivot(index='disease', columns='speakertype', values='n_total')

    all_emotions = CORE_EMOTIONS + ['other']
    emotion_to_num = {e: i for i, e in enumerate(all_emotions)}
    matrix_numeric = emotion_pivot.map(
        lambda x: emotion_to_num.get(x, -1) if pd.notna(x) else np.nan
    )

    dis_rows = emotion_pivot.index.tolist()
    spk_cols = emotion_pivot.columns.tolist()

    cmap = ListedColormap([EMOTION_COLORS[e] for e in all_emotions])
    fig, ax = plt.subplots(figsize=(14, max(7, len(dis_rows) * 0.7)))
    im = ax.imshow(matrix_numeric.values, cmap=cmap, aspect='auto',
                   vmin=0, vmax=len(all_emotions) - 1, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, ticks=range(len(all_emotions)))
    cbar.set_label('Dominant Emotion', rotation=270, labelpad=25, fontsize=10)
    cbar.ax.set_yticklabels(all_emotions, fontsize=9)

    ax.set_xticks(range(len(spk_cols)))
    ax.set_xticklabels(spk_cols, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(dis_rows)))
    ax.set_yticklabels(dis_rows, fontsize=10)
    ax.set_xlabel('Speaker Type', fontsize=11)
    ax.set_ylabel('Disease', fontsize=11)
    ax.set_title(
        'Dominant Emotion by Disease x Speaker Type\n'
        'Cell annotations: emotion (% | n_emotion/n_total)',
        fontsize=12, fontweight='bold', pad=15,
    )

    for i, dis in enumerate(dis_rows):
        for j, spk in enumerate(spk_cols):
            emotion = emotion_pivot.iloc[i, j]
            pct     = pct_pivot.iloc[i, j]
            n_core  = n_core_pivot.iloc[i, j]
            n_total = n_total_pivot.iloc[i, j]
            if pd.isna(emotion) or n_core == 0:
                if n_total > 0:
                    ax.text(j, i, '0%', ha='center', va='center',
                            fontsize=7, color='grey', style='italic')
                else:
                    ax.text(j, i, '-', ha='center', va='center',
                            fontsize=8, color='lightgrey')
            else:
                label = f"{emotion}\n{pct:.1f}"
                is_dark = emotion in ['anger', 'disgust', 'sadness', 'other']
                tc = 'white' if is_dark else 'black'
                weight = 'bold' if pct > 0.4 and n_core >= 15 else 'normal'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=7.2, color=tc, fontweight=weight)

    for x in np.arange(-0.5, len(spk_cols), 1):
        ax.axvline(x, color='white', linewidth=1.5)
    for y in np.arange(-0.5, len(dis_rows), 1):
        ax.axhline(y, color='white', linewidth=1.5)

    plt.tight_layout()
    return fig


def render(df: pd.DataFrame) -> None:
    st.subheader('Dominant Emotion by Disease x Speaker Type')
    emotion_df = compute_emotion_real(df)

    emotion_pivot = emotion_df.pivot(index='disease', columns='speakertype', values='dominant_emotion')
    pct_pivot     = emotion_df.pivot(index='disease', columns='speakertype', values='dominant_pct')
    n_core_pivot  = emotion_df.pivot(index='disease', columns='speakertype', values='n_core')
    n_total_pivot = emotion_df.pivot(index='disease', columns='speakertype', values='n_total')

    dis_rows = emotion_pivot.index.tolist()
    spk_cols = emotion_pivot.columns.tolist()

    all_emotions = CORE_EMOTIONS + ['other']
    echarts_data = []
    label_grid = [['-'] * len(spk_cols) for _ in range(len(dis_rows))]

    for i, dis in enumerate(dis_rows):
        for j, spk in enumerate(spk_cols):
            emot    = emotion_pivot.iloc[i, j]
            pct     = pct_pivot.iloc[i, j]
            n_core  = int(n_core_pivot.iloc[i, j])
            n_total = int(n_total_pivot.iloc[i, j])

            if pd.isna(emot) or n_core == 0:
                val = -1
                label_grid[i][j] = '0%' if n_total > 0 else '-'
            else:
                val = float(all_emotions.index(emot))
                label_grid[i][j] = f"{emot} {pct:.1f}"

            echarts_data.append([j, i, val])

    pieces = [
        {'min': i - 0.5, 'max': i + 0.5, 'color': EMOTION_COLORS[e], 'label': e}
        for i, e in enumerate(all_emotions)
    ]
    pieces.append({'min': -1.5, 'max': -0.5, 'color': '#dddddd', 'label': 'none'})

    lg_json = json.dumps(label_grid)
    options = {
        "tooltip": {
            "position": "top",
            "formatter": JsCode(f"function(p){{var g={lg_json};return g[p.data[1]][p.data[0]];}}")
        },
        "grid": {"left": "15%", "right": "5%", "bottom": "18%", "top": "5%"},
        "xAxis": {
            "type": "category",
            "data": spk_cols,
            "axisLabel": {"rotate": 35},
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": dis_rows,
            "splitArea": {"show": True},
        },
        "visualMap": {
            "type": "piecewise",
            "pieces": pieces,
            "orient": "horizontal",
            "left": "center",
            "bottom": "0%",
        },
        "series": [{
            "type": "heatmap",
            "data": echarts_data,
            "label": {
                "show": True,
                "formatter": JsCode(f"function(p){{var g={lg_json};return g[p.data[1]][p.data[0]];}}")
            },
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }
    height_px = max(250, len(dis_rows) * 38 + 120)
    st_echarts(options=options, height=f"{height_px}px",
               key=f"emotion_speak_{'_'.join(dis_rows)}_{'_'.join(spk_cols)}")

    fig = _build_figure(emotion_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='emotion_real.png', mime='image/png')
