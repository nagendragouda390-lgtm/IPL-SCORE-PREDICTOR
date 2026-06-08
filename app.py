import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("final_score_predictor.pkl")
bat_enc = joblib.load("batting_team_encoder.pkl")
bowl_enc = joblib.load("bowling_team_encoder.pkl")

st.set_page_config(
    page_title="IPL Final Score Predictor",
    page_icon="🏏"
)

st.title("🏏 IPL Final Score Predictor")

batting_team = st.selectbox(
    "Batting Team",
    bat_enc.classes_
)

bowling_team = st.selectbox(
    "Bowling Team",
    bowl_enc.classes_
)

current_over = st.number_input(
    "Current Over",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.1
)

current_runs = st.number_input(
    "Current Runs",
    min_value=0,
    value=80
)

current_wickets = st.number_input(
    "Current Wickets",
    min_value=0,
    max_value=10,
    value=2
)

if st.button("Predict Final Score"):
    bat = bat_enc.transform([batting_team])[0]
    bowl = bowl_enc.transform([bowling_team])[0]

    prediction = model.predict([
        [
            bat,
            bowl,
            current_over,
            current_runs,
            current_wickets
        ]
    ])

    st.success(
        f"Predicted Final Score: {int(round(prediction[0]))}"
)
