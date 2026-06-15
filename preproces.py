import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  # for saving encoder objects

CATEGORICAL_COLS = [
    "Gender",
    "State",
    "City",
    "Bank",
    "Account_Type",
    "Transaction_Type",
    "Merchant_Category",
    "Transaction_Device",
    "Device_Type",
]

NUMERIC_COLS = [
    "Age",
    "Transaction_Amount",
    "Account_Balance",
]

TARGET_COLUMN = "Is_Fraud"

# Load dataset
df = pd.read_csv(
    r"C:\Users\Public\studentproject(2025-2026)\federated-airfare (2) - Copy\Bank_Transaction_Fraud_Detection.csv"
)

# Keep only required columns
df = df[CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COLUMN]]

# Apply Label Encoding
label_encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # store encoder if needed later

# Save encoded data
df.to_csv("encoded.csv", index=False)

# Save label encoders as a single pickle file
joblib.dump(label_encoders, "label_encoders.pkl")

print("Encoded shape:", df.shape)
print("Label Encoding completed successfully.")
print("Label encoders saved as 'label_encoders.pkl'")


