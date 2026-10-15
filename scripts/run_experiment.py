import os
import sys
import logging
import argparse
from tabulate import tabulate

# Adjust path for execution from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.experiment_engine import ExperimentEngine

logging.basicConfig(level=logging.WARNING) # Suppress debug info for clear CLI output
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run VM Selection Comparison Experiment")
    parser.add_argument("--vms", type=int, default=25, help="Number of VMs in the environment")
    parser.add_argument("--workloads", type=int, default=100, help="Number of workloads to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    joblib_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not joblib_files:
        print("ERROR: No trained ML models found in 'models/'. Please run scripts/train_model.py first.")
        return
        
    latest_model = sorted(joblib_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"=== Milestone 8: VM Selection Comparison Experiment ===")
    print(f"Configuration: {args.vms} VMs | {args.workloads} Workloads | Seed: {args.seed}")
    print(f"ML Model: {latest_model}")
    print("Running simulations. Please wait...")
    
    engine = ExperimentEngine(ml_model_path=model_path)
    base_res, ml_res = engine.run_comparison(num_vms=args.vms, num_workloads=args.workloads, seed=args.seed)
    
    # Prepare comparison table
    metrics = [
        "allocation_successes", "allocation_failures", "overloaded_vms", 
        "cpu_balance_std", "mem_balance_std", "io_balance_std", "net_balance_std", "composite_balance_std",
        "mean_relative_quality", 
        "mean_decision_time", "median_decision_time", 
        "min_decision_time", "max_decision_time", "std_decision_time", 
        "total_sim_time"
    ]
    
    table_data = []
    for metric in metrics:
        table_data.append([
            metric.replace("_", " ").title(),
            f"{base_res[metric]:.6f}" if isinstance(base_res[metric], float) else base_res[metric],
            f"{ml_res[metric]:.6f}" if isinstance(ml_res[metric], float) else ml_res[metric]
        ])
        
    print("\n--- EXPERIMENT RESULTS ---")
    print(tabulate(table_data, headers=["Metric", "Baseline", "Random Forest (ML)"], tablefmt="grid"))
    print("\nDatabase logging complete (experiments, experiment_results, allocation_decisions).")

if __name__ == "__main__":
    main()
