# app/analyses/real_gate.py
import pandas as pd
import streamlit as st

from data import REAL, SPEAK

def compute_real_gate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_real'] = (df[REAL] == 'real_sickness').astype(int)
    rates = (df.groupby(SPEAK)['is_real']
               .agg(rate='mean', n='count')
               .sort_values('rate', ascending=False))
    return rates

def render(df: pd.DataFrame) -> None:
    from streamlit_echarts import st_echarts, JsCode

    st.subheader('Speakertype as Realsickness Gate')
    rates = compute_real_gate(df).reset_index()
    speakers = rates[SPEAK].tolist()
    values = [round(float(v), 3) for v in rates['rate'].tolist()]

    options = {
        "title": {"text": "Real-Sickness Rate by Speakertype", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "formatter": JsCode("function(p){return p[0].name+': '+(p[0].value*100).toFixed(1)+'%';}")},
        "grid": {"left": "3%", "right": "10%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value", "max": 1, "axisLabel": {"formatter": JsCode("function(v){return (v*100).toFixed(0)+'%';}") }},
        "yAxis": {"type": "category", "data": speakers},
        "series": [{
            "type": "bar",
            "data": values,
            "label": {"show": True, "position": "right",
                      "formatter": JsCode("function(p){return (p.value*100).toFixed(1)+'%';}")},
            "itemStyle": {
                "color": {
                    "type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0,
                    "colorStops": [
                        {"offset": 0, "color": "#d73027"},
                        {"offset": 0.5, "color": "#fee08b"},
                        {"offset": 1, "color": "#1a9850"},
                    ],
                }
            },
        }],
    }
    st_echarts(options=options, height="400px")
