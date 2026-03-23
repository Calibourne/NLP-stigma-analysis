# app/analyses/supl/nlp_text.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_echarts import st_echarts
from scipy.stats import entropy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from data import DIS, REAL, SPEAK, SENT, EMOT

TEXT_COL = 'tweet_text'

TASK_OPTIONS = {
    'Real Sickness (REAL)': REAL,
    'Speaker Type (SPEAK)': SPEAK,
    'Sentiment (SENT)': SENT,
    'Emotion (EMOT)': EMOT,
}

NEGATION_PAT   = re.compile(r"\b(not|no|never|without|isn't|aren't|wasn't|don't|doesn't|didn't|can't|won't)\b")
HEDGE_PAT      = re.compile(r"\b(maybe|might|could|seems?|appears?|possibly|probably|perhaps|sort of|kind of)\b")
DISTANCING_PAT = re.compile(r"\b(my (friend|sister|brother|mom|dad|mother|father|family|husband|wife|son|daughter)|someone i know|a friend)\b")
FIRST_PAT      = re.compile(r"\b(i|me|my|myself|i'm|i've|i'll|i'd)\b")
THIRD_PAT      = re.compile(r"\b(he|she|they|his|her|their|him|them)\b")

PATTERNS = {
    'negation':    NEGATION_PAT,
    'hedge':       HEDGE_PAT,
    'distancing':  DISTANCING_PAT,
    'first_person': FIRST_PAT,
    'third_person': THIRD_PAT,
}


def _clean_text(text: str, strip_hashtags: bool = False) -> str:
    t = str(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'\bLINK\b', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    if strip_hashtags:
        t = re.sub(r'#\w+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


@st.cache_data
def _compute_pmi(texts: tuple, labels: tuple, ngram_range: tuple = (1, 2),
                 min_df: int = 10, max_features: int = 10000,
                 class_min_df: int = 5, top_n: int = 20) -> dict:
    texts = list(texts)
    labels = list(labels)
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df,
                          max_features=max_features)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    n_docs = X.shape[0]
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)

    # P(w): fraction of documents containing word w
    pw = np.asarray((X > 0).sum(axis=0)).flatten() / n_docs

    result = {}
    for lbl in unique_labels:
        mask = label_arr == lbl
        n_class = mask.sum()
        if n_class == 0:
            result[lbl] = pd.DataFrame(columns=['term', 'pmi'])
            continue
        X_cls = X[mask]
        # P(w, c): fraction of all docs that have w AND belong to class c
        pw_c = np.asarray((X_cls > 0).sum(axis=0)).flatten() / n_docs
        # P(c)
        pc = n_class / n_docs
        # word must appear in >= class_min_df docs within the class
        word_count_in_class = np.asarray((X_cls > 0).sum(axis=0)).flatten()
        valid = (word_count_in_class >= class_min_df) & (pw > 0) & (pw_c > 0)
        pmi = np.full(len(vocab), np.nan)
        pmi[valid] = np.log2(pw_c[valid] / (pw[valid] * pc))
        df_pmi = pd.DataFrame({'term': vocab, 'pmi': pmi})
        df_pmi = df_pmi[df_pmi['pmi'].notna()].sort_values('pmi', ascending=False).head(top_n)
        result[lbl] = df_pmi.reset_index(drop=True)
    return result


def _render_pmi(df: pd.DataFrame):
    st.subheader('Analysis 1 - Prediction-Conditional PMI Lexicon')
    st.caption(
        'Top n-grams (1-2) with highest pointwise mutual information per predicted label, '
        'scoped to a single disease. Reveals vocabulary driving GPT label decisions per disease.'
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        task_label = st.selectbox('Task', list(TASK_OPTIONS.keys()), key='pmi_task')
    with c2:
        diseases = sorted(df[DIS].unique()) if DIS in df.columns else []
        disease  = st.selectbox('Disease', diseases, key='pmi_disease')
    with c3:
        strip_hash = st.checkbox('Strip hashtags', value=False, key='pmi_hash')

    task_col = TASK_OPTIONS[task_label]
    if task_col not in df.columns:
        st.warning(f'Column `{task_col}` not found in data.')
        return

    sub = df[df[DIS] == disease][[TEXT_COL, task_col]].dropna() if disease else df[[TEXT_COL, task_col]].dropna()
    if len(sub) < 20:
        st.warning(f'Too few tweets for {disease} ({len(sub)}). Select a larger disease group.')
        return

    texts_clean = tuple(sub[TEXT_COL].apply(lambda t: _clean_text(t, strip_hash)))
    labels      = tuple(sub[task_col].astype(str))

    with st.spinner(f'Computing PMI for {disease}...'):
        pmi_dict = _compute_pmi(texts_clean, labels, top_n=20)

    if not pmi_dict:
        st.warning('No PMI results computed.')
        return

    st.caption(f'n = {len(sub):,} tweets  |  disease: **{disease}**  |  task: **{task_label}**')
    label_cols = list(pmi_dict.keys())
    cols = st.columns(min(len(label_cols), 4))
    for i, lbl in enumerate(label_cols):
        with cols[i % len(cols)]:
            df_lbl = pmi_dict[lbl]
            if df_lbl.empty:
                st.markdown(f'**{lbl}**')
                st.caption('No qualifying terms.')
            else:
                terms = df_lbl['term'].tolist()[::-1]
                scores = [round(float(v), 3) for v in df_lbl['pmi'].tolist()[::-1]]
                st_echarts(options={
                    'title': {'text': lbl, 'textStyle': {'fontSize': 13}},
                    'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'},
                                'formatter': '{b}: {c}'},
                    'grid': {'left': '2%', 'right': '8%', 'containLabel': True},
                    'xAxis': {'type': 'value', 'name': 'PMI',
                              'nameTextStyle': {'fontSize': 10}},
                    'yAxis': {'type': 'category', 'data': terms,
                              'axisLabel': {'fontSize': 10}},
                    'series': [{
                        'type': 'bar',
                        'data': scores,
                        'itemStyle': {'color': '#5470c6'},
                        'label': {'show': True, 'position': 'right',
                                  'formatter': '{c}', 'fontSize': 9},
                    }],
                }, height='420px', key=f'pmi_chart_{lbl}_{disease}')


def _render_coherence(df: pd.DataFrame):
    st.subheader('Analysis 2 - Cross-Task Coherence Audit')
    st.caption(
        'Flags implausible label co-occurrences and shows conditional label distributions '
        'across any two tasks.'
    )

    disease = st.selectbox('Disease', sorted(df[DIS].unique()), key='dis_coh')
    sub = df[df[DIS] == disease]

    implausible = [
        (SPEAK, 'disease',        REAL,  'real_sickness',    'disease entity cannot be sick'),
        (EMOT,  'joy',            SENT,  'negative',         'joy + negative'),
        (SPEAK, 'not_conclusive', REAL,  'real_sickness',    'unclear speaker + definite sickness'),
        (EMOT,  'disgust',        SENT,  'positive',         'disgust + positive'),
        (EMOT,  'anger',          SENT,  'positive',         'anger + positive'),
    ]

    rows = []
    for col_a, val_a, col_b, val_b, desc in implausible:
        if col_a not in sub.columns or col_b not in sub.columns:
            continue
        mask = (sub[col_a] == val_a) & (sub[col_b] == val_b)
        count = mask.sum()
        rate = count / len(sub) if len(sub) > 0 else 0.0
        rows.append({'Description': desc, f'{col_a}={val_a}': '', f'{col_b}={val_b}': '',
                     'Count': count, 'Rate': rate})

    if rows:
        flag_df = pd.DataFrame([
            {'Description': r['Description'], 'Count': r['Count'], 'Rate': r['Rate']}
            for r in rows
        ])
        st.markdown('#### Flagged combinations')
        st.dataframe(
            flag_df.style.format({'Rate': '{:.2%}'}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info('No implausible pairs could be evaluated (columns missing).')

    st.markdown('#### Conditional crosstab heatmap')
    col_a_label = st.selectbox('Task A (rows)', list(TASK_OPTIONS.keys()), key='coh_a')
    col_b_label = st.selectbox('Task B (columns)', list(TASK_OPTIONS.keys()),
                               index=1, key='coh_b')
    col_a = TASK_OPTIONS[col_a_label]
    col_b = TASK_OPTIONS[col_b_label]

    if col_a not in sub.columns or col_b not in sub.columns:
        st.warning('Selected columns not found in data.')
        return
    if col_a == col_b:
        st.warning('Please select two different tasks.')
        return

    ct = pd.crosstab(sub[col_a], sub[col_b])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    x_cats = ct_norm.columns.tolist()
    y_cats = ct_norm.index.tolist()[::-1]  # reverse so top of chart = first row
    data = [[xi, yi, round(float(ct_norm.iloc[len(y_cats)-1-yi, xi]), 3)]
            for yi in range(len(y_cats)) for xi in range(len(x_cats))]
    options = {
        'tooltip': {'position': 'top'},
        'grid': {'left': '15%', 'right': '5%', 'containLabel': True},
        'xAxis': {'type': 'category', 'data': x_cats, 'axisLabel': {'rotate': 35, 'fontSize': 10}},
        'yAxis': {'type': 'category', 'data': y_cats, 'axisLabel': {'fontSize': 10}},
        'visualMap': {'min': 0, 'max': 1, 'calculable': True, 'orient': 'horizontal',
                      'left': 'center', 'bottom': '0%', 'inRange': {'color': ['#edf8fb', '#006d2c']}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 9},
                    'emphasis': {'itemStyle': {'shadowBlur': 10}}}]
    }
    st_echarts(options=options, height='400px', key='coh_heatmap')


def _render_negation(df: pd.DataFrame):
    st.subheader('Analysis 3 - Negation & Hedging Audit')
    st.caption(
        'Rates of linguistic uncertainty markers per predicted label. '
        'High negation/hedge rates in a class may indicate hedged or distanced language.'
    )

    disease = st.selectbox('Disease', sorted(df[DIS].unique()), key='dis_neg')
    task_label = st.selectbox('Task', list(TASK_OPTIONS.keys()), key='neg_task')
    task_col = TASK_OPTIONS[task_label]

    if task_col not in df.columns:
        st.warning(f'Column `{task_col}` not found in data.')
        return

    sub = df[df[DIS] == disease][[TEXT_COL, task_col]].dropna().copy()
    sub['_text_lower'] = sub[TEXT_COL].str.lower()

    for pat_name, pat in PATTERNS.items():
        sub[pat_name] = sub['_text_lower'].apply(lambda t: bool(pat.search(t)))

    labels_present = sorted(sub[task_col].dropna().unique())
    if not labels_present:
        st.warning('No labels found.')
        return

    rates = {}
    for lbl in labels_present:
        grp = sub[sub[task_col] == lbl]
        rates[lbl] = {p: grp[p].mean() for p in PATTERNS}

    rate_df = pd.DataFrame(rates).T
    rate_df.index.name = 'label'

    labels_list = rate_df.index.tolist()
    series = [
        {'name': pat, 'type': 'bar', 'data': [round(float(rate_df.loc[lbl, pat])*100, 1) for lbl in labels_list]}
        for pat in rate_df.columns
    ]
    options = {
        'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        'legend': {'bottom': 0, 'textStyle': {'fontSize': 9}},
        'grid': {'left': '3%', 'right': '4%', 'bottom': '15%', 'containLabel': True},
        'xAxis': {'type': 'category', 'data': labels_list, 'axisLabel': {'rotate': 30, 'fontSize': 10}},
        'yAxis': {'type': 'value', 'axisLabel': {'formatter': '{value}%', 'fontSize': 10}},
        'series': series,
    }
    st_echarts(options=options, height='380px', key=f'neg_chart_{task_col}_{disease}')

    if task_col == SPEAK:
        writer_mask = sub[task_col] == 'writer'
        non_writer_mask = sub[task_col] != 'writer'
        fp_writer = sub.loc[writer_mask, 'first_person'].mean() if writer_mask.any() else np.nan
        fp_other = sub.loc[non_writer_mask, 'first_person'].mean() if non_writer_mask.any() else np.nan
        st.info(
            f'**First-person rate:** writer = {fp_writer:.1%}  |  '
            f'non-writer = {fp_other:.1%}  '
            f'(expected: writer > non-writer if model captures self-reference)'
        )


def _render_kl(df: pd.DataFrame):
    st.subheader('Analysis 4 - Disease-Stratified Distribution Shift')
    st.caption(
        'KL divergence of per-disease label distribution from the corpus-wide reference. '
        'Higher KL = the model predicts differently for that disease.'
    )

    task_label = st.selectbox('Task', list(TASK_OPTIONS.keys()), key='kl_task')
    task_col = TASK_OPTIONS[task_label]

    if task_col not in df.columns or DIS not in df.columns:
        st.warning(f'Required columns not found in data.')
        return

    sub = df[[DIS, task_col]].dropna()
    all_labels = sorted(sub[task_col].unique())
    if not all_labels:
        st.warning('No labels found.')
        return

    # Corpus-wide reference distribution
    ref_counts = sub[task_col].value_counts().reindex(all_labels, fill_value=0)
    ref_dist = (ref_counts + 1e-9) / (ref_counts.sum() + 1e-9 * len(all_labels))

    diseases = sub[DIS].unique()
    records = []
    for dis in diseases:
        grp = sub[sub[DIS] == dis]
        if len(grp) < 30:
            continue
        dis_counts = grp[task_col].value_counts().reindex(all_labels, fill_value=0)
        dis_dist = (dis_counts + 1e-9) / (dis_counts.sum() + 1e-9 * len(all_labels))
        kl = float(entropy(dis_dist.values, ref_dist.values))
        records.append({'disease': dis, 'n': len(grp), 'kl': kl,
                        **{lbl: dis_dist[lbl] for lbl in all_labels}})

    if not records:
        st.warning('No diseases with n >= 30.')
        return

    kl_df = pd.DataFrame(records).sort_values('kl', ascending=False).reset_index(drop=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown('#### KL divergence (sorted)')
        st.dataframe(
            kl_df[['disease', 'n', 'kl']].rename(columns={'disease': 'Disease', 'n': 'N', 'kl': 'KL div'})
                .style.format({'KL div': '{:.4f}'}),
            use_container_width=True, hide_index=True,
        )

    with col_right:
        st.markdown('#### Label distribution per disease (sorted by KL)')
        dis_order = kl_df['disease'].tolist()
        plot_df = kl_df.set_index('disease')[all_labels].loc[dis_order]

        dis_list = plot_df.index.tolist()  # already sorted by KL descending
        series = [
            {'name': lbl, 'type': 'bar', 'stack': 'total',
             'data': [round(float(plot_df.loc[d, lbl])*100, 1) for d in dis_list],
             'label': {'show': False}}
            for lbl in plot_df.columns
        ]
        options = {
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'legend': {'bottom': 0, 'textStyle': {'fontSize': 9}},
            'grid': {'left': '12%', 'right': '5%', 'bottom': '12%', 'containLabel': True},
            'xAxis': {'type': 'value', 'axisLabel': {'formatter': '{value}%', 'fontSize': 9}, 'max': 100},
            'yAxis': {'type': 'category', 'data': dis_list[::-1], 'axisLabel': {'fontSize': 10}},
            'series': series,
        }
        st_echarts(options=options, height=f'{max(350, len(dis_list)*30)}px', key=f'kl_bar_{task_col}')


def _render_confidence(df: pd.DataFrame):
    st.subheader('Analysis 5 - Lexical Confidence Proxy')
    st.caption(
        'Average PMI score of each tweet\'s tokens against its predicted label\'s PMI dictionary. '
        'Lower score = less confident / less typical prediction.'
    )

    disease = st.selectbox('Disease', sorted(df[DIS].unique()), key='dis_conf')
    task_label = st.selectbox('Task', list(TASK_OPTIONS.keys()), key='conf_task')
    strip_hash = st.checkbox('Strip hashtags', value=False, key='conf_hash')
    bottom_n = st.slider('Bottom-N tweets to inspect', min_value=5, max_value=50,
                         value=20, key='conf_n')

    task_col = TASK_OPTIONS[task_label]
    if task_col not in df.columns:
        st.warning(f'Column `{task_col}` not found in data.')
        return

    sub = df[df[DIS] == disease][[TEXT_COL, task_col, DIS]].dropna().copy()
    if len(sub) < 20:
        st.warning(f'Too few tweets for {disease} ({len(sub)}). Select a larger disease group.')
        return

    texts_clean = tuple(sub[TEXT_COL].apply(lambda t: _clean_text(t, strip_hash)))
    labels = tuple(sub[task_col].astype(str))

    with st.spinner('Computing PMI (top 500)...'):
        pmi_dict = _compute_pmi(texts_clean, labels, top_n=500)

    # Build flat {label: {term: pmi}} for scoring
    pmi_lookup = {}
    for lbl, lbl_df in pmi_dict.items():
        if lbl_df.empty:
            pmi_lookup[lbl] = {}
        else:
            pmi_lookup[lbl] = dict(zip(lbl_df['term'], lbl_df['pmi']))

    def _score(text: str, label: str) -> float:
        lookup = pmi_lookup.get(label, {})
        if not lookup:
            return 0.0
        tokens = text.split()
        scores = [lookup.get(t, 0.0) for t in tokens]
        # Also check bigrams
        for i in range(len(tokens) - 1):
            bg = tokens[i] + ' ' + tokens[i + 1]
            scores.append(lookup.get(bg, 0.0))
        return float(np.mean(scores)) if scores else 0.0

    sub = sub.copy()
    sub['_text_clean'] = list(texts_clean)
    sub['score'] = [_score(t, l) for t, l in zip(sub['_text_clean'], sub[task_col])]

    st.markdown('#### Score distribution per label')
    all_scores = sub['score'].values
    bin_min, bin_max = float(all_scores.min()), float(all_scores.max())
    n_bins = 30
    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)
    bin_centers = [round((bin_edges[i]+bin_edges[i+1])/2, 3) for i in range(n_bins)]
    unique_labels = sorted(sub[task_col].unique())
    series = []
    for lbl in unique_labels:
        grp = sub[sub[task_col] == lbl]['score'].values
        counts, _ = np.histogram(grp, bins=bin_edges, density=True)
        series.append({'name': lbl, 'type': 'bar', 'barGap': '-100%',  # overlap bars
                       'data': [round(float(c), 4) for c in counts],
                       'itemStyle': {'opacity': 0.55}})
    options = {
        'tooltip': {'trigger': 'axis'},
        'legend': {'bottom': 0, 'textStyle': {'fontSize': 9}},
        'grid': {'left': '3%', 'right': '4%', 'bottom': '15%', 'containLabel': True},
        'xAxis': {'type': 'category', 'data': [str(b) for b in bin_centers],
                  'axisLabel': {'rotate': 30, 'fontSize': 9},
                  'name': 'Avg PMI score', 'nameLocation': 'middle', 'nameGap': 35},
        'yAxis': {'type': 'value', 'name': 'Density', 'nameTextStyle': {'fontSize': 9}},
        'series': series,
    }
    st_echarts(options=options, height='380px', key=f'conf_hist_{task_col}_{disease}')

    st.markdown(f'#### Bottom-{bottom_n} least confident tweets')
    bottom_df = (sub.nsmallest(bottom_n, 'score')
                    [[TEXT_COL, task_col, DIS, 'score']]
                    .rename(columns={TEXT_COL: 'Tweet', task_col: 'Predicted', DIS: 'Disease', 'score': 'PMI Score'}))
    st.dataframe(
        bottom_df.style.format({'PMI Score': '{:.3f}'}),
        use_container_width=True, hide_index=True,
    )


_STIGMA_TYPE = {
    'hiv': 'mixed', 'obesity': 'judgment', 'diabetic': 'judgment',
    'leprosy': 'mixed', 'hpv': 'judgment', 'tourette': 'disgust',
    'epilepsy': 'disgust', 'alzheimer': 'judgment', 'parkinson': 'disgust',
    'cancer': 'judgment', 'psoriasis': 'disgust', 'vitiligo': 'disgust',
    'asthma': 'judgment', 'fibro': 'judgment', 'celiac': 'judgment',
}
_STIGMA_PALETTE = {'disgust': '#E07B54', 'judgment': '#5A9EC9', 'mixed': '#8E44AD', 'unknown': '#aaaaaa'}

_LINKAGE_METHODS = ['ward', 'average', 'complete', 'single']
_METRIC_OPTIONS  = ['euclidean', 'cosine', 'cityblock']


def _render_superclustering(df: pd.DataFrame):
    st.subheader('Analysis 6 - Disease Super-Clustering')
    st.caption(
        'Hierarchical clustering of diseases based on their GPT-4o label distribution profiles '
        'across all four tasks. Groups diseases by behavioral similarity in the corpus, '
        'independent of the theoretical stigma taxonomy.'
    )

    tasks = [REAL, SPEAK, SENT, EMOT]
    if DIS not in df.columns or any(t not in df.columns for t in tasks):
        st.warning('Required columns not found.')
        return

    # ── Feature matrix ────────────────────────────────────────────────────────
    min_n = st.slider('Min tweets per disease', 30, 300, 100, key='clust_min_n')

    rows = []
    for dis in sorted(df[DIS].unique()):
        sub = df[df[DIS] == dis]
        if len(sub) < min_n:
            continue
        feats = {}
        for task in tasks:
            dist = sub[task].value_counts(normalize=True)
            for lbl, rate in dist.items():
                feats[f'{task.replace("_classification","")}::{lbl}'] = rate
        feats['_disease'] = dis
        feats['_n'] = len(sub)
        rows.append(feats)

    if len(rows) < 3:
        st.warning('Need at least 3 diseases with sufficient tweets to cluster.')
        return

    feat_df = pd.DataFrame(rows).set_index('_disease').drop(columns=['_n']).fillna(0)
    diseases = feat_df.index.tolist()

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    method  = c1.selectbox('Linkage method', _LINKAGE_METHODS, key='clust_method')
    metric  = c2.selectbox('Distance metric', _METRIC_OPTIONS, key='clust_metric')
    scale   = c3.checkbox('Standardize features (z-score)', value=False, key='clust_scale')

    X = feat_df.values.astype(float)
    if scale:
        X = StandardScaler().fit_transform(X)

    if metric == 'cosine' and method == 'ward':
        st.caption('Note: ward linkage requires euclidean distance - switching metric to euclidean.')
        metric = 'euclidean'

    dist_vec = pdist(X, metric=metric)
    Z = linkage(dist_vec, method=method)

    # ── Stigma-type leaf colours ───────────────────────────────────────────────
    leaf_colors = {
        dis: _STIGMA_PALETTE.get(_STIGMA_TYPE.get(dis.lower(), 'unknown'), '#aaaaaa')
        for dis in diseases
    }

    # ── Figure: dendrogram + feature heatmap side by side ────────────────────
    import seaborn as sns
    fig = plt.figure(figsize=(14, max(5, len(diseases) * 0.45 + 2)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.05)
    ax_dend = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    dend = dendrogram(
        Z, labels=diseases, orientation='left', ax=ax_dend,
        color_threshold=0, above_threshold_color='#555555',
        leaf_font_size=9,
    )
    # Colour leaf labels by stigma type
    for lbl in ax_dend.get_yticklabels():
        lbl.set_color(leaf_colors.get(lbl.get_text(), '#aaaaaa'))

    ax_dend.set_title(f'Dendrogram\n({method} / {metric})', fontsize=9)
    ax_dend.set_xlabel('Distance')
    ax_dend.tick_params(axis='x', labelsize=7)

    # Reorder heatmap rows to match dendrogram leaf order
    ordered = dend['ivl']
    heat_data = feat_df.loc[ordered]
    sns.heatmap(
        heat_data, ax=ax_heat, cmap='YlOrRd', linewidths=0.3,
        cbar_kws={'label': 'Label rate', 'shrink': 0.6},
        yticklabels=True, xticklabels=True,
    )
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=8)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    ax_heat.set_title('Label distribution profile\n(rows ordered by dendrogram)', fontsize=9)
    ax_heat.set_ylabel('')

    # Legend for stigma type colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=t.capitalize()) for t, c in _STIGMA_PALETTE.items() if t != 'unknown']
    ax_dend.legend(handles=legend_handles, title='Stigma type', fontsize=7,
                   title_fontsize=7, loc='lower left')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Feature importance ────────────────────────────────────────────────────
    with st.expander('Feature variance (which dimensions drive clustering)'):
        var_df = (pd.DataFrame({'feature': feat_df.columns,
                                'variance': feat_df.var().values})
                  .sort_values('variance', ascending=False).head(20))
        st.dataframe(var_df.style.format({'variance': '{:.4f}'}),
                     use_container_width=True, hide_index=True)


def render(df: pd.DataFrame):
    if TEXT_COL not in df.columns:
        st.warning(f'Column `{TEXT_COL}` not found in the loaded data. NLP text analyses require tweet text.')
        return

    st.subheader('NLP Text Analyses (GPT-4o Predictions)')
    st.caption('Five unsupervised NLP analyses on GPT-4o predicted labels, using tweet text only (no ground truth).')

    t1, t2, t3, t4, t5, t6 = st.tabs([
        'PMI Lexicon', 'Coherence Audit', 'Negation & Hedging',
        'Distribution Shift', 'Confidence Proxy', 'Super-Clustering',
    ])
    with t1:
        _render_pmi(df)
    with t2:
        _render_coherence(df)
    with t3:
        _render_negation(df)
    with t4:
        _render_kl(df)
    with t5:
        _render_confidence(df)
    with t6:
        _render_superclustering(df)
