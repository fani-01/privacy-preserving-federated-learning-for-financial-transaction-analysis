import flwr as fl
import numpy as np

# Define the strategy for federated learning
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Send updates to all clients
    fraction_evaluate=1.0,
    min_fit_clients=2,  # At least 2 clients should participate
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Start the federated learning server
fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
