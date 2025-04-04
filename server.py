import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import time
import logging

# Optional: Better log visibility
logging.basicConfig(level=logging.INFO)

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
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=None,
    )

# Start the Flower server
def start_server():
    logging.info("Starting Flower server on TCP port 10000 (Render-compatible)...")

    fl.server.start_server(
        server_address="0.0.0.0:10000",  # Use TCP port 10000 for Render
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=get_strategy(),
    )

    logging.info("✅ FL Training completed. Flower server execution finished.")

    # Optional: Keep alive if running as a long-lived service
    # while True:
    #     time.sleep(60)

if __name__ == "__main__":
    start_server()
