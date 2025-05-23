import streamlit as st

# Set up the app title and sidebar
st.set_page_config(page_title="Cricket AI WebApp", layout="wide")

st.title("🏏 Cricket AI Prediction Suite")
st.markdown("Welcome to the **CricketAI** platform. Select a prediction mode from the left menu.")

# Sidebar navigation
app_mode = st.sidebar.selectbox(
    "Choose a Feature",
    ["🏏 Wicket Type Prediction", "📈 Player Score Prediction"]
)

# Route to sub-apps based on selection
if app_mode == "🏏 Wicket Type Prediction":
    st.subheader("Wicket Type Prediction")
    st.info("This module will let you predict how a batter might get out based on match context.")
    # You’ll add your wicket prediction UI here later

elif app_mode == "📈 Player Score Prediction":
    st.subheader("Player Score Prediction")
    st.info("This module will estimate how many runs a batter might score based on history and match conditions.")
    # You’ll add your score prediction UI here later
