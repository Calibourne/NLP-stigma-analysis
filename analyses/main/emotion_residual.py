# app/analyses/emotion_residual.py
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts, JsCode
from scipy.stats import chi2_contingency

from data import DIS, EMOT

def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-squared standardized residuals: disease × emotion."""
    ct = pd.crosstab(df[DIS], df[EMOT])
    chi2, _, _, expected = chi2_contingency(ct)
    residuals = (ct.values - expected) / np.sqrt(expected)
    return pd.DataFrame(residuals, index=ct.index, columns=ct.columns)

def render(df: pd.DataFrame) -> None:
    st.subheader('Emotion Residual Analysis')
    res = compute_residuals(df)
    vmax = float(np.abs(res.values).max())
    diseases = list(res.index)
    emotions = list(res.columns)

    data = []
    for i, dis in enumerate(diseases):
        for j, emot in enumerate(emotions):
            data.append([j, i, round(float(res.loc[dis, emot]), 3)])

    options = {
        "title": {"text": "Chi-squared Standardized Residuals: Disease × Emotion", "left": "center"},
        "tooltip": {"position": "top", "formatter": JsCode(f"function(p){{var y={diseases};return y[p.data[1]]+' \u00d7 '+p.name+'<br/>'+p.data[2];}}")},
        "grid": {"left": "15%", "right": "10%", "bottom": "15%", "top": "10%"},
        "xAxis": {"type": "category", "data": emotions, "axisLabel": {"rotate": 30}, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": diseases, "splitArea": {"show": True}},
        "visualMap": {
            "min": -vmax, "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#2166ac", "#f7f7f7", "#d6604d"]},
        },
        "toolbox": {"feature": {"dataZoom": {}, "restore": {}}},
        "dataZoom": [{"type": "inside"}],
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "formatter": JsCode("function(p){return p.data[2];}")},
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }],
    }
    st_echarts(options=options, height="450px",
               key=f"emotion_residual_{'_'.join(emotions)}_{'_'.join(diseases)}")

    # Download as matplotlib PNG
    fig_mpl, ax = plt.subplots(figsize=(max(9, len(res.columns) * 1.1), max(5, len(res.index) * 0.7)))
    im = ax.imshow(res.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(res.columns)))
    ax.set_xticklabels(res.columns, rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(res.index)))
    ax.set_yticklabels([d.capitalize() for d in res.index], fontsize=9)
    for i in range(len(res.index)):
        for j in range(len(res.columns)):
            ax.text(j, i, f'{res.values[i,j]:.2f}', ha='center', va='center', fontsize=7.5, color='black')
    plt.colorbar(im, ax=ax, label='Standardized residual')
    ax.set_title('Chi-squared Standardized Residuals: Disease × Emotion')
    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='emotion_residual.png', mime='image/png')
