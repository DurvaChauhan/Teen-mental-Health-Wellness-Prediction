import streamlit as st
import pandas as pd
import joblib 

# Load model and encoders
model = joblib.load("models/depression_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.title("Teen Mental Health Prediction")

st.write("Predict depression risk using lifestyle and social media habits.")

# User Inputs
age = st.number_input("Age", min_value=13, max_value=19, value=16)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

daily_social_media_hours = st.slider(
    "Daily Social Media Hours",
    0.0, 15.0, 5.0
)

platform_usage = st.selectbox(
    "Platform Usage",
    ["Instagram", "TikTok", "Both"]
)

sleep_hours = st.slider(
    "Sleep Hours",
    0.0, 12.0, 7.0
)

screen_time_before_sleep = st.slider(
    "Screen Time Before Sleep (Hours)",
    0.0, 5.0, 1.0
)

academic_performance = st.slider(
    "Academic Performance",
    0, 100, 75
)

physical_activity = st.slider(
    "Physical Activity (Hours)",
    0.0, 10.0, 2.0
)

social_interaction_level = st.selectbox(
    "Social Interaction Level",
    ["low", "medium", "high"]
)

stress_level = st.slider(
    "Stress Level",
    1, 10, 5
)

anxiety_level = st.slider(
    "Anxiety Level",
    1, 10, 5
)

addiction_level = st.slider(
    "Addiction Level",
    1, 10, 5
)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_social_media_hours,
        "platform_usage": platform_usage,
        "sleep_hours": sleep_hours,
        "screen_time_before_sleep": screen_time_before_sleep,
        "academic_performance": academic_performance,
        "physical_activity": physical_activity,
        "social_interaction_level": social_interaction_level,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
        "addiction_level": addiction_level
    }])
    
    # Encode categorical columns
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.write("Encoded Input:")
    st.write(input_data)

    st.write("Prediction:", prediction)
    st.write("Probability Vector:", probability)

    risk_percent = round(probability[1] * 100, 2)

    if prediction == 1:
        st.error(
            f"Depression Risk Detected\n\nRisk Probability: {risk_percent}%"
        )
    else:
        st.success(
            f"No Depression Risk Detected\n\nRisk Probability: {risk_percent}%"
        )