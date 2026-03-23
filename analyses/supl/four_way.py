# app/analyses/four_way.py
import pandas as pd
import streamlit as st

from data import REAL, SPEAK, SENT, EMOT, COLS_4WAY

_REAL_ABB  = {'real_sickness': 'rs', 'not_real_sickness': 'nrs'}
_SPEAK_ABB = {'writer': 'wr', 'third_voice': 'tv', 'disease': 'di',
              'not_conclusive': 'nc', 'family': 'fa', 'celebrity': 'ce',
              'friends_colleagues': 'fr'}
_EMOT_ABB  = {'other': 'o', 'joy': 'j', 'anger': 'a', 'disgust': 'd',
              'fear': 'f', 'sadness': 'sa', 'surprise': 'su'}

def _abbrev(row) -> str:
    return '/'.join([_REAL_ABB.get(row[REAL], row[REAL][:3]),
                     _SPEAK_ABB.get(row[SPEAK], row[SPEAK][:2]),
                     row[SENT][:3],
                     _EMOT_ABB.get(row[EMOT], row[EMOT][:2])])

def compute_four_way(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    combo = (df.groupby(COLS_4WAY).size()
               .reset_index(name='count')
               .sort_values('count', ascending=False)
               .head(top_n)
               .reset_index(drop=True))
    combo['label'] = combo.apply(_abbrev, axis=1)
    combo['pct']   = combo['count'] / len(df) * 100
    return combo

def render(df: pd.DataFrame) -> None:
    from streamlit_echarts import st_echarts, JsCode

    st.subheader('4-Way Joint Label Combinations')
    top_n = st.slider('Show top N combos', 5, 30, 15, key='fourway_topn')
    combo = compute_four_way(df, top_n)

    labels = combo['label'].tolist()[::-1]
    counts = combo['count'].tolist()[::-1]
    pcts = [round(float(v), 1) for v in combo['pct'].tolist()[::-1]]

    options = {
        "title": {"text": f"Top-{top_n} 4-Way Label Combinations", "left": "center"},
        "tooltip": {
            "trigger": "axis", "axisPointer": {"type": "shadow"},
            "formatter": JsCode("function(p){return p[0].name+': '+p[0].value;}")
        },
        "grid": {"left": "3%", "right": "10%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value", "name": "Count"},
        "yAxis": {"type": "category", "data": labels},
        "series": [{
            "type": "bar",
            "data": counts,
            "itemStyle": {"color": "#73c0de"},
            "label": {
                "show": True, "position": "right",
                "formatter": JsCode("function(p){var pcts=" + str(pcts) + ";return p.value+' ('+pcts[p.dataIndex]+'%)';}")
            },
        }],
    }
    st_echarts(options=options, height=f"{max(300, top_n * 28)}px")
