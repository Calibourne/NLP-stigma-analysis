# app/pages/2_Supplementary.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from analyses.supl import sent_emotion, nc_bucket, real_gate, stigma_proxy, nlp_text
from filters import render_filters
import streamlit as st

st.set_page_config(page_title='Supplementary Analysis', layout='wide')
st.title('Supplementary Analysis')

if 'df_raw' not in st.session_state:
    st.warning('No data loaded. Go to Home and sync from Drive first.')
    st.stop()

df = render_filters()

tabs = st.tabs([
    'Sent × Emotion', 'Real Gate', 'NC Bucket', 'Stigma Proxies', 'NLP Text',
])
with tabs[0]: sent_emotion.render(df)
with tabs[1]: real_gate.render(df)
with tabs[2]: nc_bucket.render(df)
with tabs[3]: stigma_proxy.render(df)
with tabs[4]: nlp_text.render(df)
# with tabs[3]: inconsistency.render(df)
# with tabs[4]: four_way.render(df)
# with tabs[3]: tweet_samples.render(df)
