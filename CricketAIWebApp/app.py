# app.py — final
from pathlib import Path
import os
import json
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---TensorFlow (LSTM) guard ---
try:
    import tensorflow as tf
    from keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None
    pad_sequences = None


# --- Helpers ---
def image_fluid(path):
    """Show an image full-width across Streamlit versions; accepts Path or str."""
    path = str(path)
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)


# ---------- Page config & global styling ----------
st.set_page_config(
    page_title="Cricket AI Prediction Suite",
    page_icon="🏏",
    layout="wide",
    menu_items={"About": "CricketAI • Fan-first predictions & strategy."},
)

st.markdown(
    """
    <style>
      :root { --radius: 16px; }
      html, body, [class*="css"] { font-size: 18px; }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      .glass {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        backdrop-filter: blur(8px);
        border-radius: var(--radius);
        padding: 1.2rem;
      }
      label, .stSelectbox label, .stNumberInput label { font-weight: 600 !important; opacity: .9; }
      .stButton>button { border-radius: 999px; padding: .6rem 1.2rem; font-weight: 700; border: 1px solid rgba(0,0,0,.08); }
      .primary-btn .stButton>button { background: linear-gradient(90deg,#3b82f6,#06b6d4); color: #fff; border: none; }
      .chip { display:inline-block; padding:.25rem .7rem; border-radius:999px; background:rgba(0,0,0,.06); font-size:.9rem; margin-right:.4rem; }
      .hero { border-radius: var(--radius); padding: 1rem 1.2rem; background: linear-gradient(180deg, rgba(59,130,246,.08), rgba(6,182,212,.08)); border: 1px solid rgba(0,0,0,.05); }
      .streamlit-expanderHeader { font-weight: 700; }
      .stDataFrame { border-radius: var(--radius); overflow: hidden; }
      section[data-testid="stSidebar"] div[role="radiogroup"] label { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data & config (path-safe) ----------
BASE = Path(__file__).resolve().parent            # .../CricketAIWebApp

def find_data_path():
    # Allow compressed dataset to keep under GitHub 100MB limit
    for name in ["t20s_combined.csv.gz", "t20s_combined.parquet", "t20s_combined.csv", "t20s_combined.zip"]:
        p = BASE / "data" / name
        if p.exists():
            return p
    st.error("Dataset not found in CricketAIWebApp/data/ (expected .csv.gz / .parquet / .csv / .zip).")
    st.stop()

DATA_PATH      = find_data_path()
STRATEGY_JSON  = BASE / "wicket_prediction_model" / "strategy_mapping.json"

# LSTM (optional; can be absent on cloud)
MODELS_DIR     = BASE.parent / "models"
LSTM_MODEL     = MODELS_DIR / "lstm_model.h5"
TOKENIZER_PKL  = MODELS_DIR / "tokenizer.pkl"
LBLENC_PKL     = MODELS_DIR / "label_encoder.pkl"

# Score predictor artifacts
SCORE_MODEL    = BASE / "player_score_model" / "score_model.pkl"
SCORE_FEATURES = BASE / "player_score_model" / "feature_names.pkl"

# Header GIF
GIF_PATH       = BASE / "assets" / "stumps.gif"


# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_dropdowns(path: Path):
    # pandas auto-decompresses .gz and reads Path objects
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    batters = sorted(df["striker"].dropna().unique())
    non_strikers = sorted(df["non_striker"].dropna().unique()) if "non_striker" in df else []
    bowlers = sorted(df["bowler"].dropna().unique()) if "bowler" in df else []
    batting_teams = sorted(df["batting_team"].dropna().unique())
    bowling_teams = sorted(df["bowling_team"].dropna().unique())
    venues = sorted(df["venue"].dropna().unique())
    return df, batters, non_strikers, bowlers, batting_teams, bowling_teams, venues

try:
    dropdown_df, batters, non_strikers, bowlers, batting_teams, bowling_teams, venues = load_dropdowns(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dropdown data: {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def load_strategy_map(p: Path):
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
strategy_mapping = load_strategy_map(STRATEGY_JSON)


# ---------- Header / Hero ----------
col_logo, col_title = st.columns([1, 2], vertical_alignment="center")
with col_logo:
    if GIF_PATH.exists():
        image_fluid(GIF_PATH)
    else:
        st.markdown(
            """
            <div class="hero glass" style="text-align:center;">
              <div style="font-size:3rem;">🏏</div>
              <div style="margin-top:.25rem; opacity:.75;">Add <code>assets/stumps.gif</code> to show an animated header.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
with col_title:
    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0;">Cricket AI Prediction Suite</h1>
          <p style="margin:.35rem 0 0;">Fan-first, ball-by-ball intelligence for wicket type & player score.</p>
          <div style="margin-top:.5rem;">
            <span class="chip">T20</span>
            <span class="chip">Deep Learning</span>
            <span class="chip">Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.write("")


# ---------- Sidebar navigation ----------
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Choose a feature",
    ["🏏 Wicket Type Prediction", "📈 Player Score Prediction"],
    index=0,
)
with st.sidebar.expander("About this app", expanded=False):
    st.write(
        "Refreshed UI with larger typography, cleaner forms, and an optional header GIF. "
        "Add your GIF at `assets/stumps.gif`."
    )


# ---------- Small UI util ----------
def metric_row(cols: int = 3, items=None):
    items = items or []
    cols = st.columns(cols)
    for c, (label, value, help_) in zip(cols, items):
        with c:
            st.metric(label, value, help_)


# ---------- Wicket Type Prediction ----------
if app_mode == "🏏 Wicket Type Prediction":
    if not TF_AVAILABLE or not (LSTM_MODEL.exists() and TOKENIZER_PKL.exists() and LBLENC_PKL.exists()):
        st.warning("This feature is disabled for this deployment (TensorFlow/artifacts not available).")
    else:
        st.subheader("Wicket Type Prediction")
        st.caption("Predict the most likely dismissal type for the next delivery and show a fan-friendly tip.")

        with st.container():
            with st.form("wicket_form"):
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    batter_name = st.selectbox("Batter", batters, index=0)
                    non_striker_name = st.selectbox("Non-striker", non_strikers, index=0)
                    batting_team = st.selectbox("Batting Team", batting_teams, index=0)
                    venue = st.selectbox("Venue", venues, index=0)
                with c2:
                    bowler_name = st.selectbox("Bowler", bowlers, index=0)
                    bowling_team = st.selectbox("Bowling Team", bowling_teams, index=0)
                    runs_off_bat = st.number_input("Runs off Bat (this ball context)", min_value=0, max_value=100, step=1, value=0)
                    st.write("")

                submit = st.form_submit_button("🔮 Predict Wicket Type", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if submit:
            with st.spinner("Loading model & encoders..."):
                @st.cache_resource(show_spinner=False)
                def load_wicket_stack():
                    mdl = tf.keras.models.load_model(str(LSTM_MODEL))
                    tok = joblib.load(str(TOKENIZER_PKL))
                    le = joblib.load(str(LBLENC_PKL))
                    return mdl, tok, le
                model, tokenizer, label_encoder = load_wicket_stack()

            context_string = (
                f"Batsman: {batter_name} | Non-striker: {non_striker_name} | "
                f"Bowler: {bowler_name} | Batting Team: {batting_team} | "
                f"Bowling Team: {bowling_team} | Runs off bat: {runs_off_bat}"
            )
            seq = tokenizer.texts_to_sequences([context_string])
            padded = pad_sequences(seq, maxlen=50)

            pred_probs = model.predict(padded, verbose=0)
            pred_idx = int(np.argmax(pred_probs))
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            conf = float(np.max(pred_probs))

            metric_row(
                3,
                items=[
                    ("Predicted Wicket Type", f"{pred_label}", "Most likely dismissal"),
                    ("Confidence", f"{conf:.2%}", "Softmax probability"),
                    ("Runs Context", f"{runs_off_bat}", "User-supplied"),
                ],
            )

            st.success(f"📣 Suggested dismissal: **{pred_label}**")
            suggestion = strategy_mapping.get(str(pred_label).lower())
            if suggestion:
                with st.expander("💡 Strategy Tip"):
                    st.write(suggestion)
            else:
                st.info("No specific tip mapped for this dismissal yet.")
            with st.expander("Show model input (debug)"):
                st.code(context_string, language="text")


# ---------- Player Score Prediction ----------
elif app_mode == "📈 Player Score Prediction":
    st.subheader("Player Score Prediction")
    st.caption("Estimate a batter's expected runs this match using match-level recent form and context.")

    @st.cache_resource(show_spinner=False)
    def load_score_stack():
        model = joblib.load(str(SCORE_MODEL))
        feature_names = joblib.load(str(SCORE_FEATURES))
        return model, feature_names

    def match_level_form(df, batter: str, window: int = 5) -> float:
        """Rolling mean of TOTAL RUNS PER MATCH for this batter (previous N matches)."""
        df_b = df[df["striker"] == batter].copy()
        if df_b.empty or "match_id" not in df_b.columns:
            return 0.0
        per_match = (
            df_b.groupby(["match_id", "start_date"], dropna=False)["runs_off_bat"]
               .sum()
               .reset_index()
               .sort_values("start_date")
        )
        if per_match.empty:
            return 0.0
        per_match["form"] = per_match["runs_off_bat"].shift(1).rolling(window=window, min_periods=1).mean()
        last_form = per_match["form"].dropna().iloc[-1] if per_match["form"].notna().any() else 0.0
        return float(last_form)

    with st.container():
        with st.form("score_form_fixed"):
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                batter = st.selectbox("Batter (striker)", batters, index=0)
                batting_team = st.selectbox("Batting Team", batting_teams, index=0)
                venue = st.selectbox("Venue", venues, index=0)
            with c2:
                bowling_team = st.selectbox("Bowling Team", bowling_teams, index=0)
                recent_matches = st.slider("Recent matches for form (rolling mean)", 1, 10, 5)
            go = st.form_submit_button("🎯 Predict Score", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if go:
        try:
            model, feature_names = load_score_stack()
        except Exception as e:
            st.error(
                "Could not load the score model. Ensure "
                "`player_score_model/score_model.pkl` and "
                "`player_score_model/feature_names.pkl` exist and match.\n\n"
                f"Details: {e}"
            )
            st.stop()

        bf = match_level_form(dropdown_df, batter, window=recent_matches)

        input_df = pd.DataFrame([{
            "striker": batter,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "batter_form": bf,
        }])

        enc = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)

        try:
            raw_pred = float(model.predict(enc)[0])
            final_pred = int(np.clip(round(raw_pred), 0, 150))  # clamp to 0..150
            metric_row(
                3,
                items=[
                    ("Predicted Score", f"{final_pred} runs", "Rounded"),
                    ("Form (avg runs over last N matches)", f"{bf:.1f}", f"N={recent_matches}"),
                    ("Venue", venue, "Context"),
                ],
            )
            with st.expander("Show Encoded Input (debug)"):
                st.dataframe(enc)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Debug details"):
                st.write("Input row:")
                st.dataframe(input_df)
                st.write("Encoded columns expected by the model:")
                st.write(feature_names)
