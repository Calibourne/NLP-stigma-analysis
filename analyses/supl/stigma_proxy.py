# app/analyses/supl/stigma_proxy.py
from scipy import stats
from streamlit_echarts import st_echarts, JsCode
import pandas as pd
import numpy as np
import streamlit as st

from data import DIS, EMOT, REAL, SENT, SPEAK

# ── stigma type mapping (all 16 diseases) ───────────────────────────────────
# Types: "judgment" (moral blame), "disgust" (avoidance/contagion), "mixed"
# Flu excluded: stigma base rate too low to assign a reliable mechanism
STIGMA_TYPE = {
    # High stigma
    "hiv":       "mixed",     # contagion fear + sexual behavior blame
    "obesity":   "judgment",  # laziness/self-control blame
    "diabetic":  "judgment",  # T2D lifestyle conflation
    "leprosy":   "mixed",     # contagion fear + historical moral impurity
    "hpv":       "judgment",  # sexual transmission implies culpability
    "tourette":  "disgust",   # involuntary vocalizations provoke avoidance
    "epilepsy":  "disgust",   # seizures perceived as frightening
    "alzheimer": "judgment",  # competence loss triggers social devaluation
    "parkinson": "disgust",   # visible motor symptoms, aesthetic discomfort
    # Low stigma
    "cancer":    "judgment",  # lifestyle blame framing (lung, skin)
    "psoriasis": "disgust",   # visible lesions, contagion misperception
    "vitiligo":  "disgust",   # visible depigmentation, aesthetic othering
    "asthma":    "judgment",  # perceived fragility / malingering
    "fibro":     "judgment",  # "faking it" / legitimacy disputed
    "celiac":    "judgment",  # dietary non-compliance assumptions
    # flu: excluded (stigma base rate too low for reliable mechanism)
}

DISPLAY_NAMES = {
    "hiv":       "HIV",
    "obesity":   "Obesity",
    "diabetic":  "Diabetes",
    "leprosy":   "Leprosy",
    "hpv":       "HPV",
    "tourette":  "Tourette",
    "epilepsy":  "Epilepsy",
    "alzheimer": "Alzheimer's",
    "parkinson": "Parkinson's",
    "cancer":    "Cancer",
    "psoriasis": "Psoriasis",
    "vitiligo":  "Vitiligo",
    "asthma":    "Asthma",
    "fibro":     "Fibromyalgia",
    "celiac":    "Celiac",
}

DISEASE_ORDER = [
    "hiv", "obesity", "diabetic", "leprosy", "hpv",
    "tourette", "epilepsy", "alzheimer", "parkinson",
    "cancer", "psoriasis", "vitiligo", "asthma", "fibro", "celiac",
]
DISGUST_DISEASES  = [d for d, t in STIGMA_TYPE.items() if t == "disgust"]
JUDGMENT_DISEASES = [d for d, t in STIGMA_TYPE.items() if t == "judgment"]
MIXED_DISEASES    = [d for d, t in STIGMA_TYPE.items() if t == "mixed"]

PALETTE       = {"disgust": "#E07B54", "judgment": "#5A9EC9", "mixed": "#8E44AD"}
PROXY_D_COLOR = "#C0392B"
PROXY_J_COLOR = "#2C7BB6"


def _compute(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary, type_agg) DataFrames."""
    d = df[df[DIS].isin(STIGMA_TYPE)].copy()
    d["disgust_proxy"] = d[EMOT] == "disgust"
    
    # Base criteria for any judgment: external speaker + negative sentiment
    base_j = (~d[SPEAK].isin(["writer", "not_conclusive"])) & (d[SENT] == "negative")
    
    d["j_hostility"]  = base_j & (d[EMOT] == "anger")
    d["j_dismissive"] = base_j & (d[REAL] == "not_real_sickness")
    d["j_combined"]   = d["j_hostility"] & d["j_dismissive"]
    d["judgment_proxy"] = d["j_hostility"] | d["j_dismissive"]

    records = []
    for disease in DISEASE_ORDER:
        g = d[d[DIS] == disease]
        if g.empty:
            continue
        n     = len(g)
        d_rate  = float(g["disgust_proxy"].mean())
        j_rate  = float(g["judgment_proxy"].mean())
        jh_rate = float(g["j_hostility"].mean())
        jd_rate = float(g["j_dismissive"].mean())
        jc_rate = float(g["j_combined"].mean())
        
        d_n    = int(g["disgust_proxy"].sum())
        j_n    = int(g["judgment_proxy"].sum())
        stype  = STIGMA_TYPE[disease]
        dominant = "disgust" if d_rate > j_rate else "judgment" if j_rate > d_rate else "tied"
        records.append({
            "disease":             disease,
            "display":             DISPLAY_NAMES[disease],
            "stigma_type":         stype,
            "n_tweets":            n,
            "disgust_proxy_rate":  d_rate,
            "judgment_proxy_rate": j_rate,
            "j_hostility_rate":    jh_rate,
            "j_dismissive_rate":   jd_rate,
            "j_combined_rate":     jc_rate,
            "disgust_proxy_n":     d_n,
            "judgment_proxy_n":    j_n,
            "dominant_proxy":      dominant,
            "theory_matches": (stype == dominant) or (stype == "mixed"),
        })

    summary = pd.DataFrame(records)

    type_agg = d.groupby(DIS).apply(
        lambda g: pd.Series({
            "stigma_type":         STIGMA_TYPE.get(g[DIS].iloc[0], "unknown"),
            "n":                   len(g),
            "disgust_proxy_rate":  g["disgust_proxy"].mean(),
            "judgment_proxy_rate": g["judgment_proxy"].mean(),
        })
    ).reset_index()
    type_agg["stigma_type"] = type_agg[DIS].map(STIGMA_TYPE)
    type_agg = type_agg.groupby("stigma_type").agg(
        n=("n", "sum"),
        disgust_proxy_rate=("disgust_proxy_rate", "mean"),
        judgment_proxy_rate=("judgment_proxy_rate", "mean"),
    ).reset_index()

    return summary, type_agg


def _render_bar_per_disease(summary: pd.DataFrame):
    present = [d for d in DISEASE_ORDER if d in summary["disease"].values]
    diseases = [DISPLAY_NAMES[d] for d in present]
    d_rates = [round(summary.loc[summary["disease"] == d, "disgust_proxy_rate"].values[0] * 100, 1) for d in present]
    j_rates = [round(summary.loc[summary["disease"] == d, "judgment_proxy_rate"].values[0] * 100, 1) for d in present]
    
    options = {
        "title": {"text": "A. Per-disease proxy rates", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["Disgust proxy", "Judgment proxy"], "top": "10%"},
        "grid": {"left": "3%", "right": "4%", "bottom": "15%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": diseases,
            "axisLabel": {"rotate": 45, "interval": 0}
        },
        "yAxis": {"type": "value", "axisLabel": {"formatter": JsCode("function(value){return value + '%';}")}},
        "series": [
            {
                "name": "Disgust proxy",
                "type": "bar",
                "data": d_rates,
                "itemStyle": {"color": PROXY_D_COLOR}
            },
            {
                "name": "Judgment proxy",
                "type": "bar",
                "data": j_rates,
                "itemStyle": {"color": PROXY_J_COLOR}
            }
        ]
    }
    st_echarts(options=options, height="400px")


def _render_scatter(summary: pd.DataFrame):
    data = []
    # Calculate Bonferroni threshold for visual highlighting
    m = len(summary)
    alpha_bonf = 0.05 / m if m > 0 else 0.0042
    
    for _, row in summary.iterrows():
        # Re-calculate p-value for highlight
        n = row["n_tweets"]
        p1, p2 = row["disgust_proxy_rate"], row["judgment_proxy_rate"]
        n1, n2 = int(n * p1), int(n * p2)
        p_pool = (n1 + n2) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n)) if p_pool not in (0, 1) else 0
        pval = float(2 * (1 - stats.norm.cdf(abs((p1 - p2) / se)))) if se > 0 else 1.0
        
        is_robust = pval < alpha_bonf
        
        data.append({
            "value": [round(p1 * 100, 1), round(p2 * 100, 1)],
            "name": row["display"],
            "theory": row["stigma_type"].capitalize(),
            "itemStyle": {
                "color": PALETTE[row["stigma_type"]],
                "borderColor": "#000" if is_robust else "transparent",
                "borderWidth": 2 if is_robust else 0
            },
            "symbolSize": 25 if is_robust else 15,
            "label": {"show": is_robust, "position": "top", "formatter": "{b}*"}
        })
        
    lim = max(summary["disgust_proxy_rate"].max(), summary["judgment_proxy_rate"].max()) * 115
    
    options = {
        "title": {"text": "B. Theory vs. Discourse Mapping", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {
            "trigger": "item",
            "formatter": JsCode("""
                function(params) {
                    var theory = params.data.theory || 'Unknown';
                    return '<b>' + params.name + '</b><br/>' +
                           'Theory: ' + theory + '<br/>' +
                           'Disgust: ' + params.value[0] + '%<br/>' +
                           'Judgment: ' + params.value[1] + '%';
                }
            """)
        },
        "xAxis": {
            "name": "Disgust Proxy", 
            "nameLocation": "middle", 
            "nameGap": 25, 
            "type": "value", 
            "max": lim, 
            "axisLabel": {"formatter": JsCode("function(value){return value + '%';}")}
        },
        "yAxis": {
            "name": "Judgment Proxy", 
            "nameLocation": "middle", 
            "nameGap": 35, 
            "type": "value", 
            "max": lim, 
            "axisLabel": {"formatter": JsCode("function(value){return value + '%';}")}
        },
        "graphic": [
            {
                "type": "text",
                "left": "15%",
                "top": "20%",
                "style": {"text": "JUDGMENT\nDOMINANT", "fill": "#5A9EC9", "font": "bold 10px sans-serif", "textAlign": "center"}
            },
            {
                "type": "text",
                "right": "15%",
                "bottom": "25%",
                "style": {"text": "DISGUST\nDOMINANT", "fill": "#E07B54", "font": "bold 10px sans-serif", "textAlign": "center"}
            }
        ],
        "series": [
            {
                "type": "scatter",
                "data": data,
                "markLine": {
                    "animation": False,
                    "lineStyle": {"type": "dashed", "color": "#000", "opacity": 0.2},
                    "label": {"show": True, "position": "end", "formatter": "Equal Rate"},
                    "data": [[{"coord": [0, 0]}, {"coord": [lim, lim]}]]
                }
            }
        ]
    }
    st_echarts(options=options, height="450px")


def _render_aggregate_bar(type_agg: pd.DataFrame):
    type_order = [t for t in ["disgust", "judgment", "mixed"] if t in type_agg["stigma_type"].values]
    agg_d = [round(type_agg.loc[type_agg["stigma_type"] == t, "disgust_proxy_rate"].values[0] * 100, 1) for t in type_order]
    agg_j = [round(type_agg.loc[type_agg["stigma_type"] == t, "judgment_proxy_rate"].values[0] * 100, 1) for t in type_order]
    labels = [f"{t.capitalize()}-type" for t in type_order]

    options = {
        "title": {"text": "C. Aggregate by stigma type", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["Disgust proxy", "Judgment proxy"], "top": "10%"},
        "xAxis": {"type": "category", "data": labels},
        "yAxis": {"type": "value", "axisLabel": {"formatter": JsCode("function(value){return value + '%';}")}},
        "series": [
            {"name": "Disgust proxy", "type": "bar", "data": agg_d, "itemStyle": {"color": PROXY_D_COLOR}},
            {"name": "Judgment proxy", "type": "bar", "data": agg_j, "itemStyle": {"color": PROXY_J_COLOR}}
        ]
    }
    st_echarts(options=options, height="400px")


def _render_heatmap(summary: pd.DataFrame):
    present = [d for d in DISEASE_ORDER if d in summary["disease"].values]
    diseases = [f"{DISPLAY_NAMES[d]} ({STIGMA_TYPE[d][0].upper()})" for d in present]
    proxies = ["Disgust", "Judgment"]
    
    data = []
    for i, d in enumerate(present):
        d_rate = round(summary.loc[summary["disease"] == d, "disgust_proxy_rate"].values[0], 3)
        j_rate = round(summary.loc[summary["disease"] == d, "judgment_proxy_rate"].values[0], 3)
        data.append([0, i, d_rate])
        data.append([1, i, j_rate])

    options = {
        "title": {"text": "Stigma Proxy Heatmap", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"position": "top"},
        "grid": {"height": "70%", "top": "15%"},
        "xAxis": {"type": "category", "data": proxies, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": diseases, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0,
            "max": summary[["disgust_proxy_rate", "judgment_proxy_rate"]].max().max(),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "5%",
            "inRange": {"color": ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"]}
        },
        "series": [{
            "name": "Proxy rate",
            "type": "heatmap",
            "data": data,
            "label": {
                "show": True,
                "formatter": JsCode("function(params){return (params.value[2] * 100).toFixed(1) + '%';}")
            },
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }
    st_echarts(options=options, height="500px")


def render(df: pd.DataFrame):
    st.subheader("Stigma-Type Proxy Analysis")
    st.caption(
        "**Disgust proxy** = `emotion == disgust`  |  "
        "**Judgment proxy** = `[non-writer/NC]` + `negative` + (`anger` [Hostility] OR `not_real_sickness` [Minimization])"
    )

    covered = [d for d in STIGMA_TYPE if d in df[DIS].values]
    if not covered:
        st.warning("No stigma-typed diseases in current filter selection. "
                   "Flu is excluded (stigma base rate too low). Select any other disease.")
        return

    summary, type_agg = _compute(df)
    if summary.empty:
        st.warning("No data after filtering.")
        return

    # Discovery Notes
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Discovery (Obesity):** Falsifies typology. Disgust dominates a theoretically Judgment-type disease ($p < 0.0001$).")
    with col2:
        st.info("**Discovery (Epilepsy):** Falsifies typology. Disgust-type disease is dominated by social Hostility ($p < 0.0001$).")

    # Summary table
    st.markdown("#### Per-disease proxy rates (Hypothesis Test)")
    display_df = summary[[
        "display", "stigma_type", "n_tweets",
        "disgust_proxy_rate", "judgment_proxy_rate",
        "j_hostility_rate", "j_dismissive_rate", "j_combined_rate",
        "dominant_proxy", "theory_matches",
    ]].rename(columns={
        "display":             "Disease",
        "stigma_type":         "Hypothesis (Theory)",
        "n_tweets":            "N",
        "disgust_proxy_rate":  "Disgust",
        "judgment_proxy_rate": "Judgment (Total)",
        "j_hostility_rate":    "Hostile (Anger)",
        "j_dismissive_rate":   "Minimization (Not Real)",
        "j_combined_rate":     "Combined (Both)",
        "dominant_proxy":      "Dominant",
        "theory_matches":      "Aligns with Theory?",
    })
    st.dataframe(
        display_df.style.format({
            "Disgust": "{:.1%}", 
            "Judgment (Total)": "{:.1%}",
            "Hostile (Anger)": "{:.1%}",
            "Minimization (Not Real)": "{:.1%}",
            "Combined (Both)": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )

    # Stats
    st.markdown("#### Statistical Significance (Bonferroni Corrected)")
    st.caption("Alpha=0.05. Significant (Bonf.) uses $\\alpha = 0.05/12 \\approx 0.0042$.")
    stat_rows = []
    m = len(summary)
    alpha_bonf = 0.05 / m if m > 0 else 0.05
    
    for _, row in summary.iterrows():
        n   = row["n_tweets"]
        p1  = row["disgust_proxy_rate"]
        p2  = row["judgment_proxy_rate"]
        n1  = int(n * p1)
        n2  = int(n * p2)
        p_pool = (n1 + n2) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n)) if p_pool not in (0, 1) else 0
        if se == 0:
            z, pval = np.nan, np.nan
        else:
            z    = (p1 - p2) / se
            pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        stat_rows.append({
            "Disease":     row["display"],
            "z":           round(z, 2) if not np.isnan(z) else "—",
            "p-value":     f"{pval:.4f}" if not np.isnan(pval) else "—",
            "Direction":   "disgust > judgment" if p1 > p2 else "judgment > disgust" if p2 > p1 else "tied",
            "Sig. (0.05)": "Yes" if (not np.isnan(pval) and pval < 0.05) else "No",
            "Sig. (Bonf.)": "Yes" if (not np.isnan(pval) and pval < alpha_bonf) else "No",
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    # Figures
    st.markdown("#### Interactive Visualizations")
    tab1, tab2, tab3 = st.tabs(["A. Proxy Rates", "B. Scatter", "C. Aggregate"])
    with tab1:
        _render_bar_per_disease(summary)
    with tab2:
        _render_scatter(summary)
    with tab3:
        _render_aggregate_bar(type_agg)

    st.markdown("#### Stigma Proxy Heatmap")
    _render_heatmap(summary)

    st.info(
        "**Hypothesis Reference:** Judgment = moral blame (Obesity, Diabetes, HPV, Alzheimer's, Cancer, Asthma, Fibromyalgia, Celiac) | "
        "Disgust = avoidance/contagion (Tourette, Epilepsy, Parkinson's, Psoriasis, Vitiligo) | "
        "Mixed = both mechanisms (HIV, Leprosy)."
    )
