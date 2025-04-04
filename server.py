import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import time

# Define metric aggregation for FedAvg
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregation function for weighted metrics averaging"""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {}
    for num_examples, metric_dict in metrics:
        for metric_name, metric_value in metric_dict.items():
            weighted_sum = weighted_metrics.get(metric_name, 0.0)
            weighted_metrics[metric_name] = weighted_sum + (num_examples * metric_value)
    
    return {
        metric_name: metric_sum / total_examples 
        for metric_name, metric_sum in weighted_metrics.items()
    }

# Define the FedAvg strategy
def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=None,
    )

# Start the Flower server
def start_server():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=get_strategy(),
    )
    
    print("Training completed. Server is now idle and running...")

    # Keep the server running indefinitely
    while True:
        time.sleep(60)  # Sleep 60 seconds (adjust if needed)

if __name__ == "__main__":
    start_server()
