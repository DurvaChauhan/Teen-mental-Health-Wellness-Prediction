import pandas as pd

df = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")

print("Gender:")
print(df["gender"].unique())

print("\nPlatform:")
print(df["platform_usage"].unique())

print("\nSocial Interaction:")
print(df["social_interaction_level"].unique())