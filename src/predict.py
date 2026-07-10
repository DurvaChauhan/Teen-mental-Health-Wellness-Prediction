import joblib
import pandas as pd

# Load saved model and encoders
model = joblib.load("models/depression_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Example student data
sample = {
    "age": 17,
    "gender": "female",
    "daily_social_media_hours": 8,
    "platform_usage": "Instagram",
    "sleep_hours": 5,
    "screen_time_before_sleep": 3,
    "academic_performance": 60,
    "physical_activity": 1,
    "social_interaction_level": "low",  # may need adjustment too
    "stress_level": 8,
    "anxiety_level": 9,
    "addiction_level": 8
}
# Create DataFrame
df = pd.DataFrame([sample])

# Encode categorical features
for col in encoders:
    print(f"\nColumn: {col}")
    print("Allowed values:", list(encoders[col].classes_))
    print("Input value:", df[col].iloc[0])

    df[col] = encoders[col].transform(df[col])

# Predict
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0]

print("=" * 50)
print("Prediction Result")
print("=" * 50)

if prediction == 1:
    print("High Depression Risk")
else:
    print("Low Depression Risk")

print(f"\nProbability of No Depression: {probability[0]:.2%}")
print(f"Probability of Depression: {probability[1]:.2%}")

