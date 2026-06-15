import torch
import torch.nn as nn
import numpy as np

# ------------------ Model ------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# ------------------ Load Federated Model ------------------
def load_federated_model(model_path, input_size):
    model = SimpleNN(input_size)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model

# ------------------ Prediction ------------------
def predict_fraud(model, input_data):
    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    prediction = "Fraud" if prob >= 0.5 else "Not Fraud"
    confidence = round(prob * 100, 2)

    return prediction, confidence
