# app/analyses/sent_emotion.py
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts, JsCode

from data import SENT, EMOT, SENT_LABELS, EMOT_LABELS
from filters import get_order, apply_order

def compute_sent_emotion(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df[SENT], df[EMOT], normalize='index')
    return ct.reindex(index=[l for l in SENT_LABELS if l in ct.index],
                      columns=[l for l in EMOT_LABELS if l in ct.columns],
                      fill_value=0)

def render(df: pd.DataFrame) -> None:
    st.subheader('Sentiment × Emotion Coupling')
    ct = compute_sent_emotion(df)
    sentiments = list(ct.index)
    emotions = list(ct.columns)

    sentiments = apply_order(sentiments, get_order('sent'))
    emotions = apply_order(emotions, get_order('emot'))

    data = []
    for i, sent in enumerate(sentiments):
        for j, emot in enumerate(emotions):
            data.append([j, i, round(float(ct.loc[sent, emot]), 3)])

    options = {
        "title": {"text": "Row-normalised Sentiment × Emotion Co-occurrence", "left": "center"},
        "tooltip": {"position": "top"},
        "grid": {"left": "15%", "right": "5%", "bottom": "15%", "top": "12%"},
        "xAxis": {"type": "category", "data": emotions, "axisLabel": {"rotate": 30}, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": sentiments, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0, "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#eaf3fb", "#2196f3", "#0d47a1"]},
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
    st_echarts(options=options, height="350px")

    fig, ax = plt.subplots(
        figsize=(max(6, len(emotions) * 1.2), max(4, len(sentiments) * 0.9))
    )
    im = ax.imshow(ct.loc[sentiments, emotions].values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(sentiments)))
    ax.set_yticklabels(sentiments, fontsize=9)

    for i, sent in enumerate(sentiments):
        for j, emot in enumerate(emotions):
            value = float(ct.loc[sent, emot])
            text_color = 'white' if value >= 0.6 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label='Row-normalized proportion')
    ax.set_title('Row-normalised Sentiment & Emotion Co-occurrence')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='sent_emotion.png', mime='image/png')
