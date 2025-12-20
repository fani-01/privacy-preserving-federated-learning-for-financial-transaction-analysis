import flwr as fl
import numpy as np
from typing import List, Tuple, Optional
from flwr.common import FitRes, Parameters, parameters_to_ndarrays


# --------------------------------------------
# Custom Strategy to Save Aggregated Weights
# --------------------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):

    # This function is called after every federated round
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:

        # Run the original FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Save aggregated parameters (federated weights)
        if aggregated_params is not None:
            # Convert aggregated parameters into NumPy arrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_params)

            # Save to file
            filename = f"federated_round_{server_round}.npy"
            np.save(filename, aggregated_ndarrays, allow_pickle=True)

            print(f"✅ Saved aggregated federated weights to: {filename}")

        return aggregated_params, metrics


# --------------------------------------------
# Start Federated Learning Server
# --------------------------------------------
def main():
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Start the FL server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy
    )


if __name__ == "__main__":
    main()
