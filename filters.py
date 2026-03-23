# app/filters.py
import streamlit as st
from data import DIS, REAL, SPEAK, SENT, EMOT, REAL_LABELS, SPEAK_LABELS, SENT_LABELS, EMOT_LABELS

TEXT_COL = 'tweet_text'

# Persistent session_state keys (not widget keys - survive page navigation)
_K_DIS    = '_fv_diseases'
_K_REAL   = '_fv_real'
_K_SPEAK  = '_fv_speak'
_K_SENT   = '_fv_sent'
_K_EMOT   = '_fv_emot'
_K_MODELS = '_fv_models'
_K_MIN_WC = '_fv_min_wc'


def render_filters():
    """Render sidebar filters and return the filtered DataFrame.

    Requires 'df_raw' to be present in st.session_state (set by Home.py).
    Returns None if no data is loaded yet.
    """
    df = st.session_state.get('df_raw')
    if df is None:
        return None

    diseases = sorted(df[DIS].dropna().unique())
    models   = sorted(df['model'].unique()) if 'model' in df.columns else None

    # Reset
    if st.session_state.pop('_reset_filters', False):
        st.session_state[_K_DIS]    = diseases
        st.session_state[_K_REAL]   = REAL_LABELS
        st.session_state[_K_SPEAK]  = SPEAK_LABELS
        st.session_state[_K_SENT]   = SENT_LABELS
        st.session_state[_K_EMOT]   = EMOT_LABELS
        st.session_state[_K_MIN_WC] = False
        if models:
            st.session_state[_K_MODELS] = models

    # Initialize on first load
    st.session_state.setdefault(_K_DIS,    diseases)
    st.session_state.setdefault(_K_REAL,   REAL_LABELS)
    st.session_state.setdefault(_K_SPEAK,  SPEAK_LABELS)
    st.session_state.setdefault(_K_SENT,   SENT_LABELS)
    st.session_state.setdefault(_K_EMOT,   EMOT_LABELS)
    st.session_state.setdefault(_K_MIN_WC, False)
    if models:
        st.session_state.setdefault(_K_MODELS, models)

    with st.sidebar:
        st.divider()
        st.header('Filters')

        sel_diseases = st.pills('Diseases', options=diseases,
                                default=st.session_state[_K_DIS],
                                selection_mode='multi', key='w_diseases')
        st.session_state[_K_DIS] = sel_diseases or diseases

        with st.expander('Label filters'):
            sel_real = st.pills('Realsickness', options=REAL_LABELS,
                                default=st.session_state[_K_REAL],
                                selection_mode='multi', key='w_real')
            st.session_state[_K_REAL] = sel_real or REAL_LABELS

            sel_speak = st.pills('Speakertype', options=SPEAK_LABELS,
                                 default=st.session_state[_K_SPEAK],
                                 selection_mode='multi', key='w_speak')
            st.session_state[_K_SPEAK] = sel_speak or SPEAK_LABELS

            sel_sent = st.pills('Sentiment', options=SENT_LABELS,
                                default=st.session_state[_K_SENT],
                                selection_mode='multi', key='w_sent')
            st.session_state[_K_SENT] = sel_sent or SENT_LABELS

            sel_emot = st.pills('Emotion', options=EMOT_LABELS,
                                default=st.session_state[_K_EMOT],
                                selection_mode='multi', key='w_emot')
            st.session_state[_K_EMOT] = sel_emot or EMOT_LABELS

            if models:
                sel_models = st.pills('Models', options=models,
                                      default=st.session_state[_K_MODELS],
                                      selection_mode='multi', key='w_models')
                st.session_state[_K_MODELS] = sel_models or models
            else:
                sel_models = None

        st.divider()
        sel_min_wc = st.checkbox('Exclude tweets ≤ 2 words',
                                 value=st.session_state[_K_MIN_WC],
                                 key='w_min_wc')
        st.session_state[_K_MIN_WC] = sel_min_wc

        if st.button('Reset filters', use_container_width=True):
            st.session_state['_reset_filters'] = True
            st.rerun()

    filtered = df[
        df[DIS].isin(st.session_state[_K_DIS]) &
        df[REAL].isin(st.session_state[_K_REAL]) &
        df[SPEAK].isin(st.session_state[_K_SPEAK]) &
        df[SENT].isin(st.session_state[_K_SENT]) &
        df[EMOT].isin(st.session_state[_K_EMOT])
    ]
    if sel_models:
        filtered = filtered[filtered['model'].isin(st.session_state[_K_MODELS])]
    if st.session_state[_K_MIN_WC] and TEXT_COL in filtered.columns:
        filtered = filtered[filtered[TEXT_COL].str.split().str.len() > 2]

    return filtered
