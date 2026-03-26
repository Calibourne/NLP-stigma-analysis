# app/Home.py
import streamlit as st
from streamlit_echarts import st_echarts
from data import sync_drive, load_data, DIS, REAL
from filters import render_filters

st.set_page_config(page_title='NLP Results Analysis', layout='wide')
st.title('NLP Results Analysis')
st.caption('GPT 4o health-related tweets annotations')

# ── Drive sync ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Data')
    if st.button('Clear cache', width='stretch'):
        st.cache_data.clear()
        st.rerun()
    if st.button('Sync from Google Drive', width='stretch'):
        with st.spinner('Syncing…'):
            sync_drive()
            st.cache_data.clear()
        st.success('Synced. Reloading data…')
        st.rerun()

# ── Load + cache in session_state ──────────────────────────────────────────
try:
    df = load_data()
    st.session_state['df_raw'] = df
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Sidebar filters (shared across all pages) ────────────────────────────────
filtered = render_filters()
st.session_state['df'] = filtered

# ── Summary stats ──────────────────────────────────────────────────────────
st.subheader('Dataset Overview')
if len(filtered) < len(df):
    st.caption(f'Filters active: {len(filtered):,} / {len(df):,} tweets shown.')

col1, col2 = st.columns(2)
col1.metric('Total tweets',  len(filtered))
col2.metric('Diseases',      filtered[DIS].nunique())

st.divider()

col_a, col_b = st.columns(2)

def render_bar(df_series, title, color="#5470c6", sort_by_index=False):
    counts = df_series.value_counts()
    if sort_by_index:
        counts = counts.sort_index()
    else:
        counts = counts.sort_values(ascending=False)

    # Capitalize labels for better display
    labels = [str(i).replace('_', ' ').capitalize() for i in counts.index]
    options = {
        "title": {"text": title},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "value"},
        "series": [{"data": counts.values.tolist(), "type": "bar", "itemStyle": {"color": color}}],
    }
    st_echarts(options=options, height="400px")

with col_a:
    render_bar(filtered[DIS], "Tweets per disease")
with col_b:
    if 'tweet_text' in filtered.columns:
        # Calculate word count by splitting on whitespace
        word_counts = filtered['tweet_text'].str.split().str.len()
        render_bar(word_counts, "Word count distribution", color="#FDAA3E", sort_by_index=True)

st.download_button(
    label='Download filtered data as CSV',
    data=filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_data.csv',
    mime='text/csv',
)