import flwr as fl
from typing import Dict, List, Tuple
import os

# Aggregation function
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute weighted average of metrics from clients."""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {}
    
    for num_examples, metric_dict in metrics:
        for metric_name, metric_value in metric_dict.items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += num_examples * metric_value
    
    return {
        metric_name: metric_sum / total_examples
        for metric_name, metric_sum in weighted_metrics.items()
    }

# Flower Strategy
def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

# Entry Point
if __name__ == "__main__":
    print("🚀 Starting Flower server...")

    # Railway sets PORT environment variable
    port = int(os.environ.get("PORT", 8080))

    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=get_strategy(),
    )
