import pandas as pd

CATEGORICAL_COLS = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]

NUMERIC_COLS = [
    "duration",
    "days_left",
]

TARGET_COLUMN = "price"

df = pd.read_csv("airfare_full.csv")  # your original file

# keep only needed columns
df = df[CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COLUMN]]

# one hot encode all categorical cols on full data
df_encoded = pd.get_dummies(
    df,
    columns=CATEGORICAL_COLS,
    drop_first=False,
)

df_encoded.to_csv("airfare_encoded.csv", index=False)
print("encoded shape:", df_encoded.shape)
