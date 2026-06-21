# app/filters.py
import streamlit as st
from streamlit_sortables import sort_items
from data import (DIS, REAL, SPEAK, SENT, EMOT,
                  REAL_LABELS, SPEAK_LABELS, SENT_LABELS, EMOT_LABELS,
                  DIS_ORDER, DIS_CATEGORIES, DIS_CAT_ORDER, DIS_TO_CAT)

TEXT_COL = 'tweet_text'

_K_DIS        = '_fv_diseases'
_K_ORD_DIS    = '_ord_dis'
_K_ORD_CAT    = '_ord_cat'
_K_ORD_SPEAK  = '_ord_speak'
_K_ORD_SENT   = '_ord_sent'
_K_ORD_EMOT   = '_ord_emot'
_K_REAL       = '_fv_real'
_K_SPEAK      = '_fv_speak'
_K_SENT       = '_fv_sent'
_K_EMOT       = '_fv_emot'
_K_MODELS     = '_fv_models'
_K_MIN_WC     = '_fv_min_wc'
_K_CAT_MODE   = '_fv_cat_mode'
_K_CAT_STYLE  = '_fv_cat_style'   # 'collapse' | 'separate'
_K_CAT_FILTER = '_fv_cat_filter'  # selected category groups for filtering

_ORD_DEFAULTS = {
    'dis':   DIS_ORDER,
    'cat':   DIS_CAT_ORDER,
    'speak': SPEAK_LABELS,
    'sent':  SENT_LABELS,
    'emot':  EMOT_LABELS,
}


def get_order(var: str) -> list:
    """Return the current user-defined axis order for a variable.

    In category mode:
      collapse  → returns 4 category names
      separate  → returns individual diseases re-ordered by category block
    """
    if var == 'dis' and st.session_state.get(_K_CAT_MODE, False):
        cat_order = st.session_state.get(_K_ORD_CAT, DIS_CAT_ORDER)
        if st.session_state.get(_K_CAT_STYLE, 'collapse') == 'collapse':
            return cat_order
        # separate: individual diseases, grouped by category block
        dis_order = st.session_state.get(_K_ORD_DIS, DIS_ORDER)
        result = []
        for cat in cat_order:
            cat_diseases = DIS_CATEGORIES.get(cat, [])
            result.extend([d for d in dis_order if d in cat_diseases])
        cat_all = {d for ds in DIS_CATEGORIES.values() for d in ds}
        result.extend([d for d in dis_order if d not in cat_all])
        return result
    return st.session_state.get(f'_ord_{var}', _ORD_DEFAULTS[var])


def apply_order(items: list, canonical: list) -> list:
    """Apply canonical ordering to items; unknowns appended at end."""
    s_items = set(items)
    s_canon = set(canonical)
    return [v for v in canonical if v in s_items] + [v for v in items if v not in s_canon]


def _order_editor(storage_key: str, default_order: list):
    current = st.session_state.get(storage_key, list(default_order))
    new_order = sort_items(current, key=f'_sort_{storage_key}')
    st.session_state[storage_key] = new_order


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
        st.session_state[_K_DIS]        = diseases
        st.session_state[_K_REAL]       = REAL_LABELS
        st.session_state[_K_SPEAK]      = SPEAK_LABELS
        st.session_state[_K_SENT]       = SENT_LABELS
        st.session_state[_K_EMOT]       = EMOT_LABELS
        st.session_state[_K_MIN_WC]     = False
        st.session_state[_K_CAT_MODE]   = False
        st.session_state[_K_CAT_STYLE]  = 'collapse'
        st.session_state[_K_CAT_FILTER] = list(DIS_CAT_ORDER)
        st.session_state[_K_ORD_DIS]    = list(DIS_ORDER)
        st.session_state[_K_ORD_CAT]    = list(DIS_CAT_ORDER)
        st.session_state[_K_ORD_SPEAK]  = list(SPEAK_LABELS)
        st.session_state[_K_ORD_SENT]   = list(SENT_LABELS)
        st.session_state[_K_ORD_EMOT]   = list(EMOT_LABELS)
        if models:
            st.session_state[_K_MODELS] = models

    # Initialize on first load
    st.session_state.setdefault(_K_DIS,        diseases)
    st.session_state.setdefault(_K_REAL,       REAL_LABELS)
    st.session_state.setdefault(_K_SPEAK,      SPEAK_LABELS)
    st.session_state.setdefault(_K_SENT,       SENT_LABELS)
    st.session_state.setdefault(_K_EMOT,       EMOT_LABELS)
    st.session_state.setdefault(_K_MIN_WC,     False)
    st.session_state.setdefault(_K_CAT_MODE,   False)
    st.session_state.setdefault(_K_CAT_STYLE,  'collapse')
    st.session_state.setdefault(_K_CAT_FILTER, list(DIS_CAT_ORDER))
    st.session_state.setdefault(_K_ORD_DIS,    list(DIS_ORDER))
    st.session_state.setdefault(_K_ORD_CAT,    list(DIS_CAT_ORDER))
    st.session_state.setdefault(_K_ORD_SPEAK,  list(SPEAK_LABELS))
    st.session_state.setdefault(_K_ORD_SENT,   list(SENT_LABELS))
    st.session_state.setdefault(_K_ORD_EMOT,   list(EMOT_LABELS))
    if models:
        st.session_state.setdefault(_K_MODELS, models)

    with st.sidebar:
        st.divider()
        st.header('Filters')

        # ── Cluster filter ────────────────────────────────────────────────────
        sel_cats = st.pills(
            'Disease clusters', options=DIS_CAT_ORDER,
            default=st.session_state[_K_CAT_FILTER],
            selection_mode='multi', key='w_cat_filter',
        )
        st.session_state[_K_CAT_FILTER] = sel_cats or DIS_CAT_ORDER

        # Diseases available = those whose cluster is selected + any uncategorised
        selected_cats = set(st.session_state[_K_CAT_FILTER])
        cat_allowed = (
            [d for d in diseases if DIS_TO_CAT.get(d) in selected_cats] +
            [d for d in diseases if d not in DIS_TO_CAT]
        )

        sel_diseases = st.pills(
            'Diseases', options=cat_allowed,
            default=[d for d in st.session_state[_K_DIS] if d in cat_allowed],
            selection_mode='multi', key='w_diseases',
        )
        st.session_state[_K_DIS] = sel_diseases or cat_allowed

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

        with st.expander('Variable order'):
            st.caption('Drag to reorder axes in all heatmaps.')
            cat_mode = st.toggle('Group diseases by category',
                                 value=st.session_state[_K_CAT_MODE],
                                 key='w_cat_mode')
            st.session_state[_K_CAT_MODE] = cat_mode
            if cat_mode:
                cat_style = st.radio(
                    'Category display',
                    options=['collapse', 'separate'],
                    format_func=lambda x: 'Collapse to category' if x == 'collapse' else 'Keep separate',
                    index=0 if st.session_state[_K_CAT_STYLE] == 'collapse' else 1,
                    horizontal=True,
                    key='w_cat_style',
                    label_visibility='collapsed',
                )
                st.session_state[_K_CAT_STYLE] = cat_style
                st.markdown('**Disease categories**')
                _order_editor(_K_ORD_CAT, DIS_CAT_ORDER)
                if cat_style == 'separate':
                    st.markdown('**Diseases** *(within-category order)*')
                    _order_editor(_K_ORD_DIS, DIS_ORDER)
            else:
                st.markdown('**Diseases**')
                _order_editor(_K_ORD_DIS, DIS_ORDER)
            st.markdown('**Speakertype**')
            _order_editor(_K_ORD_SPEAK, SPEAK_LABELS)
            st.markdown('**Sentiment**')
            _order_editor(_K_ORD_SENT, SENT_LABELS)
            st.markdown('**Emotion**')
            _order_editor(_K_ORD_EMOT, EMOT_LABELS)

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

    if st.session_state[_K_CAT_MODE] and st.session_state[_K_CAT_STYLE] == 'collapse':
        filtered = filtered.copy()
        filtered[DIS] = filtered[DIS].map(DIS_TO_CAT).fillna('other')

    return filtered
