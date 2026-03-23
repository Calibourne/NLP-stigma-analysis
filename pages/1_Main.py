# app/pages/1_Main.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from app.analyses.main import speaker_real, inconsistency, discourse, emotion_residual, speakertype_residual, tweet_samples, emotion_speak, variance
from app.filters import render_filters
import streamlit as st

st.set_page_config(page_title='Main Analysis', layout='wide')
st.title('Main Text Analysis')

if 'df_raw' not in st.session_state:
    st.warning('No data loaded. Go to Home and sync from Drive first.')
    st.stop()

df = render_filters()

tabs = st.tabs([
    'Variance',
    'Inconcistencies', 
    'Discourse', 
    'Tweet Samples', 
    'Speaker x Real', 
    'Emotion x Speaker', 
    'Emotion Residuals', 
    'Speakertype Residuals', 
])
with tabs[0]: variance.render(df)
with tabs[1]: inconsistency.render(df)
with tabs[2]: discourse.render(df)
with tabs[3]: tweet_samples.render(df)
with tabs[4]: speaker_real.render(df)
with tabs[5]: emotion_speak.render(df)
with tabs[6]: emotion_residual.render(df)
with tabs[7]: speakertype_residual.render(df)
