import streamlit as st
import joblib
import os
st.write(os.listdir("."))

model = joblib.load("ipl_score_predictor.pkl")
bat_enc = joblib.load("batting_team_encoder.pkl")
bowl_enc = joblib.load("bowling_team_encoder.pkl")

st.title("IPL Score Predictor")

batting_team = st.selectbox("Batting Team", bat_enc.classes_)
bowling_team = st.selectbox("Bowling Team", bowl_enc.classes_)

over = st.number_input("Over", 0, 20)
runs = st.number_input("Runs", 0)
wickets = st.number_input("Wickets", 0, 10)

if st.button("Predict"):
    bat = bat_enc.transform([batting_team])[0]
    bowl = bowl_enc.transform([bowling_team])[0]

    score = model.predict([[bat, bowl, over, runs, wickets]])

    st.success(f"Predicted Final Score: {int(score[0])}")
