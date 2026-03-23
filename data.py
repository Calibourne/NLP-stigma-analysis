# app/data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import gdown

# Directory containing CSVs — always relative to this file, regardless of cwd
_DATA_DIR = Path(__file__).parent / 'dataset'

# ── Column names ──────────────────────────────────────────────────────────────
DIS   = 'disease_name'
REAL  = 'realsickness_classification'
SPEAK = 'speakertype_classification'
SENT  = 'sentiment_classification'
EMOT  = 'emotion_classification'

# ── Label sets ────────────────────────────────────────────────────────────────
REAL_LABELS  = ['real_sickness', 'not_real_sickness']
SPEAK_LABELS = ['writer', 'third_voice', 'disease', 'not_conclusive',
                'family', 'celebrity', 'friends_colleagues']
SENT_LABELS  = ['positive', 'neutral', 'negative']
EMOT_LABELS  = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'other']

NEG_EMOTIONS = ['anger', 'disgust', 'fear', 'sadness']

VALID_LABELS = {
    REAL:  set(REAL_LABELS),
    SPEAK: set(SPEAK_LABELS),
    SENT:  set(SENT_LABELS),
    EMOT:  set(EMOT_LABELS),
}

COLS_4WAY = [REAL, SPEAK, SENT, EMOT]

DRIVE_FOLDER_URL = st.secrets['DRIVE_FOLDER_URL']

MILD_FIXES = {
    'normalization': {
        SPEAK: {'non_conclusive': 'not_conclusive'},
        EMOT: {
            'frustration': 'anger',
            'bitterness': 'anger',
            'self-disgust': 'disgust',
            'doubt': 'fear',
            'concern': 'fear',
            'skepticism': 'fear',
            'hope': 'joy',
            'gratitude': 'joy',
            'inspiration': 'joy',
            'encouragement': 'joy',
            'admiration': 'joy',
            'compassion': 'joy',
            'support': 'joy',
            'empathy': 'joy',
            'inspire': 'joy',
            'disappointment': 'sadness',
            'pain': 'sadness',
            'guilt': 'sadness',
            'regret': 'sadness',
            'confusion': 'surprise',
            'disbelief': 'surprise',
            'sarcasm': 'other',
            'sarcastic': 'other',
            'determination': 'other',
            'embarrassment': 'other',
            'denial': 'other',
            'neutral': 'other',
        },
        SENT: {
            'sarcasm': 'negative',
            'sarcastic': 'negative',
            'anger': 'negative',
            'fear': 'negative',
            'sadness': 'negative',
            'support': 'positive',
            'humorous': 'positive',
            'surprise': 'neutral',
            'confusion': 'neutral',
            'mixed': 'neutral',
            'non_conclusive': 'neutral',
        },
    },
    'column_swaps':  [],
}

# ── Drive sync ────────────────────────────────────────────────────────────────
def sync_drive() -> None:
    """Pull latest CSVs from Google Drive into app/data/."""
    _DATA_DIR.mkdir(exist_ok=True)
    gdown.download_folder(DRIVE_FOLDER_URL, output=str(_DATA_DIR), quiet=False)

# ── Triage ────────────────────────────────────────────────────────────────────
def triage_predictions(
    df: pd.DataFrame,
    mild_fixes: dict,
    valid_labels: dict,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply mild label fixes; rows with remaining invalid labels are flagged as
    severe and excluded from cleaned_results.

    Returns (cleaned_df, reevaluation_df, report_dict).
    """
    df = df.copy()

    # Apply normalization fixes
    for col, mapping in mild_fixes.get('normalization', {}).items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # Apply column swaps
    for col_a, col_b in mild_fixes.get('column_swaps', []):
        if col_a in df.columns and col_b in df.columns:
            df[col_a], df[col_b] = df[col_b].copy(), df[col_a].copy()

    # Identify severe rows (any classification column has an invalid label)
    severe_mask = pd.Series(False, index=df.index)
    for col, valid in valid_labels.items():
        if col in df.columns:
            severe_mask |= ~df[col].isin(valid)

    cleaned     = df[~severe_mask].copy()
    reevaluation = df[severe_mask].copy()
    report = {
        'total':         len(df),
        'clean_count':   len(cleaned),
        'severe_count':  len(reevaluation),
    }
    if verbose and report['severe_count']:
        st.text(f"[triage] {report['severe_count']} severe rows excluded.")
    return cleaned, reevaluation, report

# ── Cached loader ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Read all CSVs in app/data/, normalize column names, triage, return cleaned df."""
    files = list(_DATA_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {_DATA_DIR}. Use the Drive sync button on Home."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col=[0])
        # Normalize column names to lowercase so constants always match
        df.columns = [c.lower() for c in df.columns]
        if 'disease_name' not in df.columns and 'desease' in df.columns:
            df = df.rename(columns={'desease': 'disease_name'})
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cleaned, _, _ = triage_predictions(raw, MILD_FIXES, VALID_LABELS, verbose=True)
    return cleaned.reset_index(drop=True)
