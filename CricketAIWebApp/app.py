import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import json


# Load dataset to extract dropdown options
data_path = os.path.join("data", "t20s_combined.csv")
dropdown_df = pd.read_csv(data_path)

# Extract unique sorted options for dropdowns
batters = sorted(dropdown_df["striker"].dropna().unique())
non_strikers = sorted(dropdown_df["non_striker"].dropna().unique())
bowlers = sorted(dropdown_df["bowler"].dropna().unique())
batting_teams = sorted(dropdown_df["batting_team"].dropna().unique())
bowling_teams = sorted(dropdown_df["bowling_team"].dropna().unique())
venues = sorted(dropdown_df["venue"].dropna().unique())

# Load strategy suggestions from JSON
with open("wicket_prediction_model/strategy_mapping.json", "r") as f:
    strategy_mapping = json.load(f)


# Set up the app title and sidebar
st.set_page_config(page_title="Cricket AI WebApp", layout="wide")

st.title("\U0001F3CF Cricket AI Prediction Suite")
st.markdown("Welcome to the **CricketAI** platform. Select a prediction mode from the left menu.")

# Sidebar navigation
app_mode = st.sidebar.selectbox(
    "Choose a Feature",
    ["\U0001F3CF Wicket Type Prediction", "\U0001F4C8 Player Score Prediction"]
)

# --- WICKET TYPE PREDICTION UI ---
if app_mode == "\U0001F3CF Wicket Type Prediction":
    st.subheader("Wicket Type Prediction")
    st.info("This module will let you predict how a batter might get out based on match context.")

    # Load models and encoders
    with st.spinner("Loading model and encoders..."):
        model = tf.keras.models.load_model("../models/lstm_model.h5")
        tokenizer = joblib.load("../models/tokenizer.pkl")
        label_encoder = joblib.load("../models/label_encoder.pkl")

    # Input fields
    batter_name = st.selectbox("Batsman Name", batters)
    non_striker_name = st.selectbox("Non-striker Name", non_strikers)
    bowler_name = st.selectbox("Bowler Name", bowlers)
    batting_team = st.selectbox("Batting Team", batting_teams)
    bowling_team = st.selectbox("Bowling Team", bowling_teams)
    venue = st.selectbox("Venue", venues)
    runs_off_bat = st.number_input("Runs off Bat", min_value=0, max_value=100, step=1)

    # Predict button
    if st.button("Predict Wicket Type"):
        if all([batter_name, non_striker_name, bowler_name, batting_team, bowling_team, venue]):
            context_string = f"Batsman: {batter_name} | Non-striker: {non_striker_name} | Bowler: {bowler_name} | Batting Team: {batting_team} | Bowling Team: {bowling_team} | Runs off bat: {runs_off_bat}"
            sequence = tokenizer.texts_to_sequences([context_string])
            padded_seq = pad_sequences(sequence, maxlen=50)
            pred_probs = model.predict(padded_seq)
            pred_label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]

            st.success(f"\U0001F4E2 **Predicted Wicket Type**: `{pred_label}`")

            # Dynamic strategy suggestion from JSON
            suggestion = strategy_mapping.get(pred_label.lower())

            if suggestion:
                st.warning(f"💡 Strategy: {suggestion}")
            else:
                st.info("ℹ️ Strategy suggestion not available for this dismissal.")

            print(f"Predicted Label: {pred_label}")

        else:
            st.error("⚠️ Please fill in all fields.")

# --- PLAYER SCORE PREDICTION UI (placeholder for now) ---
elif app_mode == "\U0001F4C8 Player Score Prediction":
    st.subheader("Player Score Prediction")
    st.info("This module will estimate how many runs a batter might score based on history and match conditions.")

    rf_model = joblib.load("../models/random_forest_model.pkl")

    # Inputs
    batter = st.selectbox("Select Batter", batters)
    batting_team = st.selectbox("Select Batting Team", batting_teams)
    bowling_team = st.selectbox("Select Bowling Team", bowling_teams)
    venue = st.selectbox("Select Venue", venues)
    n_matches = st.slider("Number of recent matches to consider for form", 1, 20, 5)

    if st.button("Predict Score"):
        recent_matches = dropdown_df[dropdown_df["striker"] == batter].sort_values("match_id", ascending=False)
        unique_match_ids = recent_matches["match_id"].drop_duplicates().head(n_matches)
        match_scores = []

        for mid in unique_match_ids:
            match_runs = recent_matches[recent_matches["match_id"] == mid]["runs_off_bat"].sum()
            match_scores.append(match_runs)

        batter_form = np.mean(match_scores) if match_scores else 0

        input_df = pd.DataFrame({
            "batter_form": [batter_form],
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "venue": [venue]
        })

        # Ensure categorical columns are consistent
        for col in ["batting_team", "bowling_team", "venue"]:
            input_df[col] = input_df[col].astype("category")

        try:
            predicted_score = rf_model.predict(input_df)[0]
            st.success(f"\U0001F3CF Predicted Score: `{predicted_score:.1f}` runs")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
