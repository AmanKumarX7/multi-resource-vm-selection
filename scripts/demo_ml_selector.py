import os
import sys
import logging
from datetime import datetime

# Adjust path for execution from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.environment import SimulationEnvironment
from src.simulation.workload import Workload
from src.algorithms.ml_selector import MLSelector
from src.database.repositories import ExperimentRepository, AllocationRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def demo():
    logger.info("Starting ML Selector Demonstration...")
    
    # 1. Initialize DB Experiment
    exp_name = f"ML_Demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_id = ExperimentRepository.create_experiment(name=exp_name, description="Milestone 7 ML Selector Demo")
    logger.info(f"Created Experiment DB Record: {exp_id}")
    
    # 2. Setup Environment
    env = SimulationEnvironment()
    env.generate_default_vms(count=5, cpu=100, mem=16, io=1000, net=1000)
    
    # Pre-allocate some resources to make VMs asymmetrical
    env.vms["VM_0"].allocate(90, 2, 100, 100) # Heavy CPU
    env.vms["VM_1"].allocate(10, 14, 100, 100) # Heavy Mem
    env.vms["VM_2"].allocate(50, 8, 500, 500)  # Balanced
    # VM_3 and VM_4 are completely empty
    
    # 3. Load ML Selector
    # Try to find the latest model in models directory
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    joblib_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not joblib_files:
        logger.error("No .joblib models found. Please train a model first.")
        return
        
    latest_model = sorted(joblib_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    selector = MLSelector(model_path=model_path)
    
    # 4. Generate a Test Workload
    wl = Workload(workload_id="W_TEST_1", workload_type="balanced", cpu_req=10, mem_req=2, io_req=100, net_req=100, arrival_time=0, duration=10)
    
    # 5. Evaluate Feasibility (Done by Env/Caller, not Selector)
    feasible_vms = env.get_feasible_vms(wl)
    logger.info(f"Found {len(feasible_vms)} feasible VMs.")
    
    # 6. ML Selection
    selected_vm, dt = selector.select_vm(feasible_vms, wl)
    
    if selected_vm:
        logger.info(f"ML Selector chose {selected_vm.vm_id} in {dt:.6f} seconds.")
        success = env.allocate_workload(selected_vm.vm_id, wl)
        
        # Log to DB
        # Note: We need mock internal IDs for DB tracking since we skipped full setup
        # For demo purposes, we will mock the workload and VM internal IDs as 1 and 2
        try:
            AllocationRepository.log_decision(
                experiment_id=exp_id,
                workload_internal_id=1, 
                vm_internal_id=2, 
                method="random_forest",
                decision_time=dt,
                is_successful=success
            )
            logger.info("Decision logged to database successfully.")
        except Exception as e:
            logger.warning(f"Could not log to database due to missing foreign keys (expected in demo): {e}")
    else:
        logger.warning("No VM could be selected.")
        
    logger.info("Demonstration complete.")

if __name__ == "__main__":
    demo()
