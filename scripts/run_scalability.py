import os
import sys
import logging
import argparse
import time

# Adjust path for execution from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.scalability import ScalabilityRunner

logging.basicConfig(level=logging.WARNING) # Suppress debug info for clear CLI output
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Scalability Experiment Matrices")
    parser.add_argument("--profile", type=str, choices=["quick", "standard"], default="quick", 
                        help="Profile to run: 'quick' for a smoke-test matrix, 'standard' for the full research matrix.")
    
    args = parser.parse_args()
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    joblib_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not joblib_files:
        print("ERROR: No trained ML models found in 'models/'. Please run scripts/train_model.py first.")
        return
        
    latest_model = sorted(joblib_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"=== Milestone 11: Scalability Experiments ===")
    print(f"Profile: {args.profile.upper()}")
    print(f"ML Model: {latest_model}")
    print("Running scalability matrices. This may take a while depending on the profile...")
    
    start_time = time.perf_counter()
    runner = ScalabilityRunner(ml_model_path=model_path)
    
    if args.profile == "quick":
        # A. VM-count scalability (small)
        print("Running Series A: VM-count scalability (Quick)")
        runner.run_matrix("VM_Scale_Quick", vms_list=[10, 25], workloads_list=[25], seeds=[42])
        
        # B. Workload-count scalability (small)
        print("Running Series B: Workload-count scalability (Quick)")
        runner.run_matrix("WL_Scale_Quick", vms_list=[25], workloads_list=[10, 25], seeds=[42])
        
    elif args.profile == "standard":
        # A. VM-count scalability
        print("Running Series A: VM-count scalability (Standard)")
        runner.run_matrix("VM_Scale", vms_list=[10, 25, 50, 100, 250, 500], workloads_list=[100], seeds=[42, 43, 44])
        
        # B. Workload-count scalability
        print("Running Series B: Workload-count scalability (Standard)")
        runner.run_matrix("WL_Scale", vms_list=[100], workloads_list=[25, 50, 100, 250, 500], seeds=[42, 43, 44])
        
    duration = time.perf_counter() - start_time
    print(f"\nScalability experiments completed in {duration:.2f} seconds.")
    print(f"Raw data and aggregates saved to: {runner.output_dir}")
    print("Database logging complete (experiments, experiment_results).")

if __name__ == "__main__":
    main()
