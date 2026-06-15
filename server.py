import flwr as fl
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays

# ---------------------- Model ----------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

INPUT_SIZE = 12  # 9 categorical + 3 numeric (label encoded)

# ---------------------- Custom Strategy ----------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )

        if aggregated_parameters is not None:
            # Convert Flower Parameters → NumPy arrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Load into PyTorch model
            model = SimpleNN(INPUT_SIZE)
            for param, ndarray in zip(model.parameters(), ndarrays):
                param.data = torch.tensor(ndarray, dtype=torch.float32)

            # Save global model
            torch.save(model.state_dict(), "federated_model.pth")
            print(f"[SUCCESS] Federated model saved after round {rnd}")

        return aggregated_parameters, aggregated_metrics

# ---------------------- Strategy ----------------------
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
)

# ---------------------- Start Server ----------------------
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
