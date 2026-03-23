# app/analyses/tweet_samples.py
import pandas as pd
import streamlit as st

from data import DIS, REAL, SPEAK, SENT, EMOT, COLS_4WAY

_REAL_ABB  = {'real_sickness': 'rs', 'not_real_sickness': 'nrs'}
_SPEAK_ABB = {'writer': 'wr', 'third_voice': 'tv', 'disease': 'di',
              'not_conclusive': 'nc', 'family': 'fa', 'celebrity': 'ce',
              'friends_colleagues': 'fr'}
_EMOT_ABB  = {'other': 'o', 'joy': 'j', 'anger': 'a', 'disgust': 'd',
              'fear': 'f', 'sadness': 'sa', 'surprise': 'su'}

def abbrev_combo_c(row) -> str:
    return '/'.join([_REAL_ABB.get(row[REAL], row[REAL][:3]),
                     _SPEAK_ABB.get(row[SPEAK], row[SPEAK][:2]),
                     row[SENT][:3],
                     _EMOT_ABB.get(row[EMOT], row[EMOT][:2])])

def _latex_escape(s: str) -> str:
    for char, repl in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                       ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                       ('}', r'\}'), ('~', r'\textasciitilde{}'),
                       ('^', r'\textasciicircum{}')]:
        s = s.replace(char, repl)
    return s

def compute_samples(df: pd.DataFrame, top_k: int = 5, n_tweets: int = 5) -> dict:
    """
    Returns {disease: [{'rank', 'label', 'pct', 'tweets'}, ...]} for top_k combos.
    """
    dis_labels = sorted(df[DIS].unique())
    result = {}
    for dis in dis_labels:
        sub = df[df[DIS] == dis]
        n = len(sub)
        combo = (sub.groupby(COLS_4WAY).size()
                    .reset_index(name='count')
                    .sort_values('count', ascending=False)
                    .reset_index(drop=True))
        entries = []
        for rank, (_, row) in enumerate(combo.head(top_k).iterrows(), start=1):
            mask = ((sub[REAL]  == row[REAL])  &
                    (sub[SPEAK] == row[SPEAK]) &
                    (sub[SENT]  == row[SENT])  &
                    (sub[EMOT]  == row[EMOT]))
            tweets = (sub[mask]['tweet_text']
                        .sample(min(n_tweets, int(mask.sum())), random_state=42)
                        .tolist())
            entries.append({'rank': rank, 'label': abbrev_combo_c(row),
                            'pct': row['count'] / n * 100, 'tweets': tweets})
        result[dis] = entries
    return result

def build_latex(samples: dict) -> str:
    N_COLS = 6
    mc = lambda n, t: f'\\multicolumn{{{n}}}{{@{{}}l}}{{{t}}}'

    cont_str = r'{\small\itshape (continued)}'
    cont_next_str = r'{\small\itshape (continued on next page)}'

    header = [
        r'% Auto-generated',
        r'% Requires: \usepackage{booktabs,longtable,array,csquotes}',
        r'',
        r'\begin{longtable}{@{} c l l l l p{6cm} @{}}',
        r'  \toprule',
        r'  \textbf{Rank (\%)} & \textbf{Real.} & \textbf{Speaker} & '
        r'\textbf{Sent.} & \textbf{Emotion} & \textbf{Sample tweet} \\',
        r'  \midrule',
        r'  \endfirsthead',
        '  ' + mc(N_COLS, cont_str) + ' \\\\',
        r'  \toprule',
        r'  \textbf{Rank (\%)} & \textbf{Real.} & \textbf{Speaker} & '
        r'\textbf{Sent.} & \textbf{Emotion} & \textbf{Sample tweet} \\',
        r'  \midrule',
        r'  \endhead',
        r'  \midrule',
        '  ' + mc(N_COLS, cont_next_str) + ' \\\\',
        r'  \endfoot',
        r'  \bottomrule',
        r'  \endlastfoot',
    ]
    body = []
    diseases = list(samples.keys())
    for dis in diseases:
        dis_cap = dis.capitalize()
        dis_str = r'{\textbf{\textsc{' + dis_cap + r'}}}'
        body.append('  ' + mc(N_COLS, dis_str) + ' \\\\')
        body.append(r'  \addlinespace[2pt]')
        for entry in samples[dis]:
            parts = entry['label'].split('/')
            for i, tweet in enumerate(entry['tweets']):
                rank_cell = f"{entry['rank']} ({entry['pct']:.1f}" + r'\%)' if i == 0 else ''
                cols = []
                if i == 0:
                    for p in parts:
                        cols.append(r'\texttt{' + p + r'}')
                else:
                    cols = ['', '', '', '']
                escaped_tweet = _latex_escape(str(tweet))
                t_cell = r'\enquote{' + escaped_tweet + r'}'
                if i > 0:
                    body.append(r'  \addlinespace[3pt]')
                body.append(f'  {rank_cell} & {cols[0]} & {cols[1]} & ' +
                             f'{cols[2]} & {cols[3]} & {t_cell} \\\\')
        body.append(r'  \midrule')
    if body and body[-1].strip() == r'\midrule':
        body.pop()
    return '\n'.join(header + body + [r'\end{longtable}'])

def render(df: pd.DataFrame) -> None:
    st.subheader('Sampled Tweets per Top-5 Discourse Type')
    samples = compute_samples(df)

    for dis, entries in samples.items():
        st.metric(dis.capitalize(), len(df[df[DIS]==dis]))
        # st.markdown(f'#### {dis.capitalize()} (n={len(df[df[DIS]==dis])})')
        rows = []
        for e in entries:
            for i, t in enumerate(e['tweets']):
                rows.append({'rank'   : e['rank']  if i == 0 else '',
                             'combo'  : e['label'] if i == 0 else '',
                             '%'      : f"{e['pct']:.1f}%" if i == 0 else '',
                             'tweet'  : t})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    tex = build_latex(samples)
    st.download_button(
        label='Download LaTeX table',
        data=tex.encode(),
        file_name='discourse_samples.tex',
        mime='text/plain',
    )
