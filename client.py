import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------

TARGET_COLUMN = "Is_Fraud"

# pick which dataset this client loads
CLIENT_ID = os.getenv("bank_ID", "1")   # "1", "2", "3", "4"

CSV_PATH = f"bank_{CLIENT_ID}.csv"       # your 4 encoded files


# -----------------------------
# LOAD DATA (already encoded)
# -----------------------------

def load_and_split_data(path, test_size=0.2):
    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} missing in dataset")

    y = df[TARGET_COLUMN].astype(np.float32).values
    X = df.drop(columns=[TARGET_COLUMN]).astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    return train_dataset, test_dataset


train_dataset, test_dataset = load_and_split_data(CSV_PATH)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # price output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_size = train_dataset.tensors[0].shape[1]
print(f"[Client {CLIENT_ID}] Input features:", input_size)

model = SimpleNN(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# -----------------------------
# FLOWER CLIENT
# -----------------------------
epochs=20
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in model.parameters()]

    def set_parameters(self, parameters):
        for param, new in zip(model.parameters(), parameters):
            param.data = torch.tensor(new, dtype=param.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"[Client {CLIENT_ID}] Epoch {epoch+1}/{epochs} - MSE: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"client_{CLIENT_ID}_model.pth")
        print(f"[Client {CLIENT_ID}] model saved")

        return self.get_parameters(config), len(train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()

        total_loss = 0.0
        total_abs = 0.0
        count = 0

        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                loss = criterion(preds, yb)

                bsz = yb.size(0)
                total_loss += loss.item() * bsz
                total_abs  += torch.abs(preds - yb).sum().item()
                count += bsz

        mse  = total_loss / count
        rmse = mse ** 0.5
        mae  = total_abs / count

        print(f"[Client {CLIENT_ID}] Eval - MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        return mse, len(test_dataset), {"rmse": rmse, "mae": mae}


# -----------------------------
# START CLIENT
# -----------------------------

if __name__ == "__main__":
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )



