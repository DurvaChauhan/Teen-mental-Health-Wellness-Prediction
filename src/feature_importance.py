import pandas as pd
import matplotlib.pyplot as plt
import joblib

model = joblib.load("models/depression_model.pkl")

df = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")

X = df.drop("depression_label", axis=1)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,6)
)

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png"
)

plt.show()