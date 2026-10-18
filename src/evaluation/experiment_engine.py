import time
import logging
import statistics
import numpy as np
from typing import List, Dict

from src.simulation.environment import SimulationEnvironment
from src.simulation.workload_generator import WorkloadGenerator
from src.algorithms.baseline import BaselineSelector
from src.algorithms.ml_selector import MLSelector
from src.database.repositories import ExperimentRepository, AllocationRepository, VMRepository, WorkloadRepository
from src.database.connection import get_db_connection
from src.ml.dataset_generator import DatasetGenerator

logger = logging.getLogger(__name__)

class ExperimentEngine:
    """
    Executes reproducible comparison experiments between Baseline and ML selectors.
    """
    def __init__(self, ml_model_path: str):
        self.ml_model_path = ml_model_path
        self.ground_truth_calc = DatasetGenerator(seed=0)
        
    def _create_environment(self, num_vms: int) -> SimulationEnvironment:
        env = SimulationEnvironment()
        env.generate_default_vms(num_vms, cpu=1000, mem=16, io=5000, net=1000)
        return env
        
    def _run_selector(self, env: SimulationEnvironment, workloads: List, selector, 
                      method_name: str, exp_id: int, vm_map: dict, wl_map: dict) -> dict:
        """
        Runs the simulation loop for a single selector.
        """
        total_decision_time = 0.0
        decision_times = []
        allocation_successes = 0
        allocation_failures = 0
        relative_qualities = []
        
        sim_start = time.perf_counter()
        
        # Track active workloads for expiration: list of (expiry_time, workload_id)
        active_workloads = []
        
        for wl in workloads:
            current_time = wl.arrival_time
            
            # Release expired workloads
            still_active = []
            for expiry_time, wl_id in active_workloads:
                if expiry_time <= current_time:
                    env.release_workload(wl_id)
                else:
                    still_active.append((expiry_time, wl_id))
            active_workloads = still_active
            
            # Find feasible
            feasible_vms = env.get_feasible_vms(wl)
            
            # Select (Isolated Time)
            selected_vm, dt = selector.select_vm(feasible_vms, wl)
            decision_times.append(dt)
            total_decision_time += dt
            
            # Relative Quality Metric calculation
            if feasible_vms and selected_vm:
                selected_suitability = self.ground_truth_calc.calculate_ground_truth(selected_vm, wl)
                best_suitability = max(self.ground_truth_calc.calculate_ground_truth(v, wl) for v in feasible_vms)
                
                if best_suitability > 0:
                    relative_qualities.append(selected_suitability / best_suitability)
                else:
                    relative_qualities.append(1.0 if selected_suitability == 0 else 0.0)
            
            # Allocate
            success = False
            if selected_vm:
                success = env.allocate_workload(selected_vm.vm_id, wl)
                if success:
                    active_workloads.append((current_time + wl.duration, wl.workload_id))
                    
            if success:
                allocation_successes += 1
            else:
                allocation_failures += 1
                
            # Log decision to database with real IDs (Time not counted in decision_time)
            AllocationRepository.log_decision(
                experiment_id=exp_id,
                workload_internal_id=wl_map[wl.workload_id], 
                vm_internal_id=vm_map[selected_vm.vm_id] if selected_vm else None,
                method=method_name,
                decision_time=dt,
                is_successful=success
            )

        sim_time = time.perf_counter() - sim_start
        
        # Calculate Overloaded VMs (util >= 90%)
        overloaded_vms = sum(
            1 for v in env.vms.values() 
            if v.cpu_utilization >= 90 or v.memory_utilization >= 90 or v.io_utilization >= 90 or v.network_utilization >= 90
        )
        
        # 4-Resource Balance Standard Deviations
        vms = list(env.vms.values())
        if len(vms) > 1:
            cpu_std = statistics.stdev([v.cpu_utilization for v in vms])
            mem_std = statistics.stdev([v.memory_utilization for v in vms])
            io_std = statistics.stdev([v.io_utilization for v in vms])
            net_std = statistics.stdev([v.network_utilization for v in vms])
        else:
            cpu_std = mem_std = io_std = net_std = 0.0
            
        composite_balance = (cpu_std + mem_std + io_std + net_std) / 4.0
        
        return {
            'method': method_name,
            'allocation_successes': allocation_successes,
            'allocation_failures': allocation_failures,
            'overloaded_vms': overloaded_vms,
            'cpu_balance_std': cpu_std,
            'mem_balance_std': mem_std,
            'io_balance_std': io_std,
            'net_balance_std': net_std,
            'composite_balance_std': composite_balance,
            'mean_decision_time': statistics.mean(decision_times) if decision_times else 0,
            'median_decision_time': statistics.median(decision_times) if decision_times else 0,
            'min_decision_time': min(decision_times) if decision_times else 0,
            'max_decision_time': max(decision_times) if decision_times else 0,
            'std_decision_time': statistics.stdev(decision_times) if len(decision_times) > 1 else 0,
            'total_sim_time': sim_time,
            'mean_relative_quality': statistics.mean(relative_qualities) if relative_qualities else 0.0
        }
        
    def _save_results(self, exp_id: int, results: dict):
        query = "INSERT INTO experiment_results (experiment_id, metric_name, metric_value) VALUES (%s, %s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for k, v in results.items():
                        if k == 'method':
                            continue
                        metric_name = f"{results['method']}_{k}"
                        cursor.execute(query, (exp_id, metric_name, float(v)))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
            raise

    def run_comparison(self, num_vms: int, num_workloads: int, seed: int = 42):
        """
        Runs a perfectly fair comparison between Baseline and ML.
        """
        logger.info(f"Starting comparison: {num_vms} VMs, {num_workloads} Workloads, Seed {seed}")
        
        exp_id = ExperimentRepository.create_experiment(
            name=f"Comparison_vms{num_vms}_wl{num_workloads}_s{seed}",
            description="Milestone 8 Comparison Engine"
        )
        
        # Generate the shared workload sequence
        wg = WorkloadGenerator(seed=seed)
        workloads = wg.generate_workloads(num_workloads)
        
        # --- BASELINE RUN ---
        env_baseline = self._create_environment(num_vms)
        
        # Map IDs (using Baseline as the canonical env for db creation)
        vm_map = VMRepository.insert_vms(exp_id, list(env_baseline.vms.values()))
        wl_map = WorkloadRepository.insert_workloads(exp_id, workloads)
        
        baseline_selector = BaselineSelector()
        logger.info("Running Baseline selector...")
        baseline_res = self._run_selector(env_baseline, workloads, baseline_selector, "baseline", exp_id, vm_map, wl_map)
        self._save_results(exp_id, baseline_res)
        
        # --- ML RUN ---
        env_ml = self._create_environment(num_vms)
        ml_selector = MLSelector(self.ml_model_path)
        logger.info("Running ML selector...")
        ml_res = self._run_selector(env_ml, workloads, ml_selector, "random_forest", exp_id, vm_map, wl_map)
        self._save_results(exp_id, ml_res)
        
        logger.info("Comparison complete.")
        return baseline_res, ml_res
