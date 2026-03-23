# app/analyses/inconsistency.py
import pandas as pd
import streamlit as st

from data import REAL, EMOT, SENT, SPEAK, NEG_EMOTIONS

# Rules: (description, mask_fn)

RULES = [
    ("Positive sentiment\n+ negative emotion",
     lambda df: ((df[SENT] == 'positive') & (df[EMOT].isin(NEG_EMOTIONS)))),
    ("Negative sentiment\n+ joy",
     lambda df: ((df[SENT] == 'negative') & (df[EMOT] == 'joy'))),
    ("Speaker=disease\n+ not_real_sickness",
     lambda df: ((df[SPEAK] == 'disease') & (df[REAL] == 'not_real_sickness'))),
    ("Speaker=not_conclusive\n+ real_sickness",
     lambda df: ((df[SPEAK] == 'not_conclusive') & (df[REAL] == 'real_sickness'))),
]


# RULES = [
#     ('joy + not_real_sickness',
#      lambda df: (df[EMOT] == 'joy') & (df[REAL] == 'not_real_sickness')),
#     ('positive sentiment + real_sickness',
#      lambda df: (df[SENT] == 'positive') & (df[REAL] == 'real_sickness')),
#     ('sadness + not_real_sickness',
#      lambda df: (df[EMOT] == 'sadness') & (df[REAL] == 'not_real_sickness')),
# ]

def compute_inconsistencies(df: pd.DataFrame) -> dict:
    rows = []
    flagged = pd.Series(False, index=df.index)
    for desc, mask_fn in RULES:
        mask = mask_fn(df)
        flagged |= mask
        rows.append({'rule': desc, 'count': int(mask.sum()),
                     'pct': mask.mean() * 100})
    breakdown = pd.DataFrame(rows).sort_values('count', ascending=False)
    return {'total_inconsistent': int(flagged.sum()), 'breakdown': breakdown}

def render(df: pd.DataFrame) -> None:
    from streamlit_echarts import st_echarts, JsCode

    st.subheader('Cross-Task Logical Inconsistency Audit')
    result = compute_inconsistencies(df)
    st.metric('Total inconsistent rows', result['total_inconsistent'],
              delta=f"{result['total_inconsistent']/len(df):.1%} of dataset",
              delta_color='inverse')

    bd = result['breakdown']
    rules = bd['rule'].tolist()
    counts = bd['count'].tolist()
    pcts = [round(float(v), 1) for v in bd['pct'].tolist()]

    options = {
        "title": {"text": "Inconsistency Counts by Rule", "left": "center"},
        "tooltip": {
            "trigger": "axis", "axisPointer": {"type": "shadow"},
            "formatter": JsCode("function(p){return p[0].name+': '+p[0].value;}")
        },
        "grid": {"left": "3%", "right": "10%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": rules},
        "series": [{
            "type": "bar",
            "data": counts,
            "itemStyle": {"color": "#ee6666"},
            "label": {
                "show": True, "position": "right",
                "formatter": JsCode("function(p){var pcts=" + str(pcts) + ";return p.value+' ('+pcts[p.dataIndex]+'%)';}")
            },
        }],
    }
    st_echarts(options=options, height="300px")
