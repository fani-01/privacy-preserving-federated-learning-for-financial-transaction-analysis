import pandas as pd

# load encoded dataset
df = pd.read_csv("airfare_encoded.csv")

print("Full encoded dataset shape:", df.shape)

# --- split by airline one-hot columns ---

df_air_india = df[df["airline_Air_India"] == 1]
df_vistara   = df[df["airline_Vistara"] == 1]

# save subsets
df_air_india.to_csv("airfare_1.csv", index=False)
df_vistara.to_csv("airfare_2.csv", index=False)

print("airfare_air_india.csv:", df_air_india.shape)
print("airfare_vistara.csv:", df_vistara.shape)
