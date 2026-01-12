import streamlit as st
import pandas as pd
from predict import Predictor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Personal Health Tracker",
    layout="centered"
)

st.title("Personal Health Tracker (ML-Powered)")

# ---------------- Cache Predictor ----------------
@st.cache_resource
def load_predictor():
    return Predictor()

predictor = load_predictor()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Input your measurements")

age = st.sidebar.number_input("Age", 12, 90, 30)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex_v = 1 if sex == "Male" else 0

weight_kg = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0)
height_cm = st.sidebar.number_input("Height (cm)", 120.0, 230.0, 170.0)

activity_label = st.sidebar.selectbox(
    "Activity level", ["Sedentary", "Light", "Moderate", "Active"]
)
activity_map = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3}
activity_level = activity_map[activity_label]

sleep_hour = st.sidebar.slider("Sleep hours", 0.0, 12.0, 7.0)
rest_hour = st.sidebar.number_input("Resting heart rate", 30, 140, 70)
steps_per_day = st.sidebar.number_input("Average steps per day", 0, 50000, 5000)
stress_level = st.sidebar.slider("Stress level (0 = low, 1 = high)", 0.0, 1.0, 0.3)

# ---------------- Prediction (LIVE) ----------------
row = {
    "Age": int(age),
    "Sex": int(sex_v),
    "Weight_kg": float(weight_kg),
    "Height_cm": float(height_cm),
    "Sleep_hour": float(sleep_hour),
    "Rest_hour": float(rest_hour),
    "Activity_level": int(activity_level),
    "Steps_per_day": int(steps_per_day),
    "Stress_level": float(stress_level),
}

pred = predictor.predict(row)

# ---------------- Output ----------------
st.subheader("Predictions")

st.metric(
    "Estimated daily calories",
    f"{pred['calories_needed']} kcal"
)

st.progress(int(pred["fatigue_prob"] * 100))
st.write(f"Fatigue probability: **{pred['fatigue_prob']:.2f}**")

# ---------------- Meal Recommendation ----------------
st.subheader("Meal recommendation (example)")

cal = pred["calories_needed"]
st.write(f"**Suggested daily calories:** {cal} kcal")
st.write(f"Breakfast: ~{int(cal*0.25)} kcal — oatmeal + banana + milk")
st.write(f"Lunch: ~{int(cal*0.40)} kcal — rice/roti + veg + chicken/tofu")
st.write(f"Dinner: ~{int(cal*0.35)} kcal — salad + soup + lean protein")

# ---------------- Debug (REMOVE LATER) ----------------
st.caption("Debug: current model input")
st.json(row)
