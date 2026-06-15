import pandas as pd

# load encoded dataset
df = pd.read_csv("encoded.csv")

print("Full encoded dataset shape:", df.shape)

# --- split by airline one-hot columns ---

df_sbi = df[df["Bank"] == 0]
df_HDFC   = df[df["Bank"] == 1]
df_ICICI   = df[df["Bank"] == 2]
df_BOB   = df[df["Bank"] == 3]

# save subsets
df_sbi.to_csv("bank_1.csv", index=False)
df_HDFC.to_csv("bank_2.csv", index=False)
df_ICICI.to_csv("bank_3.csv", index=False)
df_BOB.to_csv("bank_4.csv", index=False)
print("bank_1.csv:", df_sbi.shape)
print("bank_2.csv:", df_HDFC.shape)
print("bank_3.csv:", df_ICICI.shape)
print("bank_4.csv:", df_BOB.shape)
