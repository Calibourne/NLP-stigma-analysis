# app/analyses/sent_emotion.py
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts, JsCode

from data import SENT, EMOT, SENT_LABELS, EMOT_LABELS

def compute_sent_emotion(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df[SENT], df[EMOT], normalize='index')
    return ct.reindex(index=[l for l in SENT_LABELS if l in ct.index],
                      columns=[l for l in EMOT_LABELS if l in ct.columns],
                      fill_value=0)

def render(df: pd.DataFrame) -> None:
    from streamlit_echarts import st_echarts

    st.subheader('Sentiment × Emotion Coupling')
    ct = compute_sent_emotion(df)
    sentiments = list(ct.index)
    emotions = list(ct.columns)

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
