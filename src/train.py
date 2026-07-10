import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")

# Store encoders
encoders = {}

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

model.fit(X_train_resampled, y_train_resampled)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "models/depression_model.pkl")

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel saved successfully!")
print("Model path: models/depression_model.pkl")
print("Encoder path: models/encoders.pkl")