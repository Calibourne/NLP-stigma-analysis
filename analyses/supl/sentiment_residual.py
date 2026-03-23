# app/analyses/sentiment_residual.py
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from streamlit_echarts import JsCode, st_echarts

from app.data import DIS, SENT, SENT_LABELS


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-squared standardized residuals: disease × sentiment."""
    ct = pd.crosstab(df[DIS], df[SENT])
    _, _, _, expected = chi2_contingency(ct)
    residuals = (ct.values - expected) / np.sqrt(expected)
    return pd.DataFrame(residuals, index=ct.index, columns=ct.columns)


def render(df: pd.DataFrame) -> None:
    st.subheader('Sentiment Residual Analysis')
    res = compute_residuals(df)
    # Order columns by SENT_LABELS
    cols = [c for c in SENT_LABELS if c in res.columns]
    res = res[cols]
    vmax = float(np.abs(res.values).max())
    diseases = list(res.index)
    sentiments = list(res.columns)

    data = []
    for i, dis in enumerate(diseases):
        for j, sent in enumerate(sentiments):
            data.append([j, i, round(float(res.loc[dis, sent]), 3)])

    options = {
        "title": {"text": "Chi-squared Standardized Residuals: Disease × Sentiment", "left": "center"},
        "tooltip": {"position": "top", "formatter": JsCode(f"function(p){{var y={diseases};return y[p.data[1]]+' \u00d7 '+p.name+'<br/>'+p.data[2];}}")},
        "grid": {"left": "15%", "right": "10%", "bottom": "15%", "top": "12%"},
        "xAxis": {"type": "category", "data": sentiments, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": diseases, "splitArea": {"show": True}},
        "visualMap": {
            "min": -vmax, "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#2166ac", "#f7f7f7", "#d6604d"]},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "formatter": JsCode("function(p){return p.data[2];}")},
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }],
    }
    st_echarts(options=options, height="450px")

    # Download as matplotlib PNG
    fig, ax = plt.subplots(figsize=(max(6, len(sentiments) * 1.5), max(5, len(diseases) * 0.7)))
    im = ax.imshow(res.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sentiments)))
    ax.set_xticklabels(sentiments, fontsize=9)
    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels([d.capitalize() for d in diseases], fontsize=9)
    for i in range(len(diseases)):
        for j in range(len(sentiments)):
            ax.text(j, i, f'{res.values[i,j]:.2f}', ha='center', va='center', fontsize=8, color='black')
    plt.colorbar(im, ax=ax, label='Standardized residual')
    ax.set_title('Chi-squared Standardized Residuals: Disease × Sentiment')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='sentiment_residual.png', mime='image/png')
