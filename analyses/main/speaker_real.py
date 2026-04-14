# app/analyses/speaker_real.py
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_echarts import st_echarts, JsCode

from data import DIS, REAL, SPEAK

def compute_speaker_real(df: pd.DataFrame) -> pd.DataFrame:
    """Real-sickness rate per (speakertype, disease). Returns pivot table."""
    df = df.copy()
    df['is_real'] = (df[REAL] == 'real_sickness').astype(float)
    pivot = df.groupby([SPEAK, DIS])['is_real'].mean().unstack(DIS)
    return pivot

def render(df: pd.DataFrame) -> None:
    st.subheader('Speakertype × Realsickness Joint Distribution')
    pivot = compute_speaker_real(df)
    diseases = [c for c in pivot.columns]
    speakers = list(pivot.index)

    # ECharts heatmap data: [x_idx, y_idx, value]
    data = []
    for i, spk in enumerate(speakers):
        for j, dis in enumerate(diseases):
            val = pivot.loc[spk, dis]
            data.append([j, i, round(float(val), 3) if not pd.isna(val) else "-"])

    options = {
        "tooltip": {"position": "top", "formatter": JsCode(f"function(p){{var y={speakers};return y[p.data[1]]+' \u00d7 '+p.name+'<br/>'+p.data[2];}}")},
        "grid": {"left": "15%", "right": "10%", "bottom": "15%", "top": "10%"},
        "xAxis": {"type": "category", "data": diseases, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": speakers, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0, "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#d73027", "#fee08b", "#1a9850"]},
        },
        "toolbox": {"feature": {"dataZoom": {}, "restore": {}}},
        "dataZoom": [{"type": "inside"}],
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "formatter": JsCode("function(p){return p.data[2];}")},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }
    st_echarts(options=options, height="450px")

    # Download as matplotlib PNG
    fig_mpl, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='black')
    plt.colorbar(im, ax=ax, label='real_sickness rate')
    ax.set_title('Real-Sickness Rate per Speakertype × Disease')
    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='speaker_real.png', mime='image/png')
