# app/analyses/main/variance.py
import io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from streamlit_echarts import st_echarts, JsCode

from data import (DIS, REAL, SPEAK, SENT, EMOT,
                      REAL_LABELS, SPEAK_LABELS, SENT_LABELS, EMOT_LABELS)

_BG_COLOR    = 'white'
_SPINE_COLOR = '#444444'
_FONT        = 'serif'

_TASK_COLORS = {
    'realsickness': '#1A5276',
    'speakertype':  '#1E8449',
    'sentiment':    '#784212',
    'emotion':      '#7D3C98',
}
_TASK_DISPLAY = {
    'realsickness': 'Real-sickness',
    'speakertype':  'Speaker type',
    'sentiment':    'Sentiment',
    'emotion':      'Emotion',
}
_TASK_LABEL_LIST = [
    (REAL,  REAL_LABELS),
    (SPEAK, SPEAK_LABELS),
    (SENT,  SENT_LABELS),
    (EMOT,  EMOT_LABELS),
]


def compute_variance(df: pd.DataFrame) -> dict:
    """Return prop_tables and cramers dicts keyed by short task name."""
    N = len(df)
    cramers = {}
    for task, _ in _TASK_LABEL_LIST:
        ct = pd.crosstab(df[DIS], df[task])
        if min(ct.shape) < 2:
            cramers[task.replace('_classification', '')] = (float('nan'), float('nan'))
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (N * (min(ct.shape) - 1)))
        cramers[task.replace('_classification', '')] = (v, p)

    prop_tables = {}
    for task, labels in _TASK_LABEL_LIST:
        key = task.replace('_classification', '')
        ct = pd.crosstab(df[DIS], df[task], normalize='index') * 100
        prop_tables[key] = ct[[c for c in labels if c in ct.columns]]

    return {'prop_tables': prop_tables, 'cramers': cramers}


def _make_cmap(hex_color: str):
    return LinearSegmentedColormap.from_list('task_cmap', ['#F8F9F9', hex_color], N=256)


def _build_figure(prop_tables: dict, dis_display: list) -> plt.Figure:
    fig = plt.figure(figsize=(15, 7), facecolor=_BG_COLOR)
    fig.suptitle('Cross-Disease Label Distributions', fontsize=15,
                 fontweight='bold', y=0.98, fontfamily=_FONT, color='#222222')

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                           left=0.08, right=0.97, top=0.88, bottom=0.12)

    for col_idx, tkey in enumerate(['realsickness', 'speakertype', 'sentiment', 'emotion']):
        ax = fig.add_subplot(gs[col_idx])
        prop   = prop_tables[tkey]
        data   = prop.values
        labels = list(prop.columns)
        n_rows, n_cols = data.shape

        ax.imshow(data, aspect='auto', cmap=_make_cmap(_TASK_COLORS[tkey]),
                  vmin=0, vmax=100, interpolation='none')

        for ri in range(n_rows):
            for ci in range(n_cols):
                val = data[ri, ci]
                tc  = 'white' if val > 58 else '#222222'
                ax.text(ci, ri, f'{val:.0f}', ha='center', va='center',
                        fontsize=8.2 if n_cols <= 3 else 7.0,
                        fontfamily=_FONT, color=tc)

        for xi in np.arange(-0.5, n_cols, 1):
            ax.axvline(xi, color='#DDDDDD', linewidth=0.5)
        for yi in np.arange(-0.5, n_rows, 1):
            ax.axhline(yi, color='#DDDDDD', linewidth=0.5)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(labels, rotation=38, ha='right',
                           fontsize=7.8 if n_cols <= 4 else 7.0, fontfamily=_FONT)
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', length=0, pad=2)

        if col_idx == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(dis_display, fontsize=9, fontfamily=_FONT)
            ax.tick_params(axis='y', length=0)
        else:
            ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE_COLOR)
            sp.set_linewidth(0.6)

        ax.set_title(_TASK_DISPLAY[tkey], fontsize=10, fontweight='bold',
                     fontfamily=_FONT, color=_TASK_COLORS[tkey], pad=8)

    fig.text(
        0.525, 0.0,
        'Label distribution (% of disease tweets) - values are row percentages within each disease',
        ha='center', va='bottom', fontsize=9, fontfamily=_FONT, color='#333333',
    )
    plt.tight_layout()
    return fig


def render(df: pd.DataFrame) -> None:
    st.subheader('Cross-Disease Label Distributions')
    result = compute_variance(df)
    prop_tables = result['prop_tables']

    dis_labels  = sorted(df[DIS].unique())
    dis_display = [d.replace('_', ' ').capitalize() for d in dis_labels]

    # ── ECharts: 4-panel heatmap ──────────────────────────────────────────────
    task_order = ['realsickness', 'speakertype', 'sentiment', 'emotion']
    cols_ui = st.columns(4)

    for col_ui, tkey in zip(cols_ui, task_order):
        with col_ui:
            prop   = prop_tables[tkey]
            data   = prop.values
            labels = list(prop.columns)
            n_rows, n_cols = data.shape

            echarts_data = [
                [ci, ri, round(float(data[ri, ci]), 1)]
                for ri in range(n_rows)
                for ci in range(n_cols)
            ]

            options = {
                "title": {
                    "text": _TASK_DISPLAY[tkey],
                    "left": "center",
                    "textStyle": {"color": _TASK_COLORS[tkey], "fontSize": 12, "fontWeight": "bold"},
                },
                "tooltip": {
                    "position": "top",
                    "formatter": JsCode("function(p){return p.name+' - '+p.data[2]+'%';}"),
                },
                "grid": {"left": "8%", "right": "4%", "bottom": "28%", "top": "12%"},
                "xAxis": {
                    "type": "category",
                    "data": labels,
                    "axisLabel": {"rotate": 38, "fontSize": 8},
                    "splitArea": {"show": True},
                },
                "yAxis": {
                    "type": "category",
                    "data": dis_display,
                    "splitArea": {"show": True},
                    "axisLabel": {"show": tkey == 'realsickness', "fontSize": 8},
                },
                "visualMap": {
                    "min": 0, "max": 100,
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": "0%",
                    "inRange": {"color": ["#F8F9F9", _TASK_COLORS[tkey]]},
                    "textStyle": {"fontSize": 9},
                },
                "series": [{
                    "type": "heatmap",
                    "data": echarts_data,
                    "label": {
                        "show": True,
                        "formatter": JsCode("function(p){return p.data[2];}"),
                        "fontSize": 8,
                    },
                    "emphasis": {"itemStyle": {"shadowBlur": 5}},
                }],
            }
            st_echarts(options=options, height="500px")

    # ── Download PNG ──────────────────────────────────────────────────────────
    fig = _build_figure(prop_tables, dis_display)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='variance.png', mime='image/png')
