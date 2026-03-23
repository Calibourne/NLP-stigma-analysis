# app/analyses/nc_bucket.py

import pandas as pd
import streamlit as st

from app.data import SPEAK, REAL, SENT, EMOT, REAL_LABELS, SENT_LABELS, EMOT_LABELS

def compute_nc_stats(df: pd.DataFrame) -> dict:
    nc  = df[df[SPEAK] == 'not_conclusive']
    all_ = df
    nc_share = len(nc) / len(df)
    rows = []
    for task, labels in [(REAL, REAL_LABELS), (SENT, SENT_LABELS), (EMOT, EMOT_LABELS)]:
        for label in labels:
            nc_prop  = (nc[task]  == label).mean() if len(nc)  else 0
            all_prop = (all_[task] == label).mean()
            rows.append({'task': task.replace('_classification',''),
                         'label': label, 'nc': nc_prop, 'overall': all_prop})
    return {'nc_share': nc_share, 'task_comparison': pd.DataFrame(rows)}

def render(df: pd.DataFrame) -> None:
    from streamlit_echarts import st_echarts

    st.subheader('not_conclusive as Uncertainty Bucket')
    stats = compute_nc_stats(df)
    st.metric('Share of not_conclusive tweets', f"{stats['nc_share']:.1%}")

    tc = stats['task_comparison']

    # One chart per task — mirrors the 3-subplot notebook layout
    for task_name, task_labels in [
        ('realsickness', REAL_LABELS),
        ('sentiment',    SENT_LABELS),
        ('emotion',      EMOT_LABELS),
    ]:
        sub = tc[tc['task'] == task_name].set_index('label').reindex(task_labels).dropna()
        present      = sub.index.tolist()
        overall_vals = [round(float(v) * 100, 1) for v in sub['overall'].tolist()]
        nc_vals      = [round(float(v) * 100, 1) for v in sub['nc'].tolist()]

        options = {
            "title": {"text": task_name.replace('_', ' ').title(), "left": "center"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": ["Overall", "not_conclusive"], "bottom": 0},
            "grid": {"left": "3%", "right": "4%", "bottom": "15%", "containLabel": True},
            "xAxis": {"type": "category", "data": present, "axisLabel": {"rotate": 35}},
            "yAxis": {"type": "value", "name": "%"},
            "series": [
                {"name": "Overall", "type": "bar", "data": overall_vals,
                 "itemStyle": {"color": "#4C72B0"}, "barGap": "0%"},
                {"name": "not_conclusive", "type": "bar", "data": nc_vals,
                 "itemStyle": {"color": "#DD8452"}},
            ],
        }
        st_echarts(options=options, height="320px")
