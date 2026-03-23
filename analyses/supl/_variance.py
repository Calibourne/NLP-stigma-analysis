# app/analyses/variance.py
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from data import DIS, REAL, SPEAK, SENT, EMOT, REAL_LABELS, SPEAK_LABELS, SENT_LABELS, EMOT_LABELS

def compute_variance(df: pd.DataFrame) -> pd.DataFrame:
    task_label_list = [
        (REAL,  REAL_LABELS),
        (SPEAK, SPEAK_LABELS),
        (SENT,  SENT_LABELS),
        (EMOT,  EMOT_LABELS),
    ]
    rows = []
    for task, labels in task_label_list:
        ct = pd.crosstab(df[DIS], df[task], normalize='index')
        for label in [c for c in labels if c in ct.columns]:
            rows.append({
                'task' : task.replace('_classification', ''),
                'label': label,
                'std'  : round(ct[label].std(), 4),
                'mean' : round(ct[label].mean(), 4),
                'min'  : round(ct[label].min(), 4),
                'max'  : round(ct[label].max(), 4),
            })
    return pd.DataFrame(rows).sort_values('std', ascending=False).reset_index(drop=True)

def render(df: pd.DataFrame) -> None:
    st.subheader('Cross-Disease Variance in Label Distributions')
    std_df = compute_variance(df)
    st.dataframe(std_df, width='stretch')

    top = std_df.head(10)
    labels = [f"{r['task']}/{r['label']}" for _, r in top.iterrows()]

    # top-10 most variable, highest std at top
    values = top['std'].tolist()

    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "8%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value", "name": "Std of proportion across diseases"},
        "yAxis": {"type": "category", "data": labels[::-1]},
        "series": [{
            "type": "bar",
            "data": values[::-1],
            "itemStyle": {"color": "#4e79a7"},
            "label": {"show": True, "position": "right", "formatter": "{c:.4f}"},
        }],
    }
    st_echarts(options=options, height="400px")

    # Download as matplotlib PNG
    fig_mpl, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels[::-1], top['std'].values[::-1], color='steelblue')
    ax.set_xlabel('Std of proportion across diseases')
    ax.set_title('Top-10 Most Disease-Variable Class Labels')
    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    st.download_button('Download as PNG', buf, file_name='variance.png', mime='image/png')
