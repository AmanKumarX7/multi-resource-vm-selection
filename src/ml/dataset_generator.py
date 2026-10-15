import json
import logging
import os
import random
import statistics
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.workload_generator import WorkloadGenerator

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generates training data for the Machine Learning VM selector.
    Simulates a sequence of workloads, evaluates all VMs (feasible and infeasible),
    and records the state (features) and ground-truth suitability (target).
    """

    WORKLOAD_ENCODING = {
        'balanced': 0,
        'cpu-intensive': 1,
        'memory-intensive': 2,
        'io-intensive': 3,
        'network-intensive': 4
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(self.seed)

    def calculate_ground_truth(self, vm: VM, wl: Workload) -> float:
        """
        Independent Suitability Target:
        Evaluates the hypothetical resultant state if the workload is placed on the VM.
        - If infeasible, score is 0.0.
        - If feasible, rewards balanced utilization and penalizes high peak usage.
        Suitability = 100.0 - max(U) - (0.1 * std_dev(U)), bounded [0, 100].
        """
        if not vm.can_allocate(wl.cpu_req, wl.mem_req, wl.io_req, wl.net_req):
            return 0.0
            
        # Calculate resulting utilizations
        cpu_u = min(100.0, ((vm.cpu_alloc + wl.cpu_req) / vm.cpu_cap * 100.0)) if vm.cpu_cap > 0 else 100.0
        mem_u = min(100.0, ((vm.mem_alloc + wl.mem_req) / vm.mem_cap * 100.0)) if vm.mem_cap > 0 else 100.0
        io_u = min(100.0, ((vm.io_alloc + wl.io_req) / vm.io_cap * 100.0)) if vm.io_cap > 0 else 100.0
        net_u = min(100.0, ((vm.net_alloc + wl.net_req) / vm.net_cap * 100.0)) if vm.net_cap > 0 else 100.0
        
        utilizations = [cpu_u, mem_u, io_u, net_u]
        max_u = max(utilizations)
        std_u = statistics.stdev(utilizations)
        
        score = 100.0 - max_u - (0.1 * std_u)
        return max(0.0, min(100.0, score)) # bound cleanly [0.0, 100.0]

    def extract_features(self, vm: VM, wl: Workload) -> dict:
        """
        Extracts pre-allocation state features strictly to avoid target leakage.
        """
        return {
            # VM States
            'vm_cpu_util': vm.cpu_utilization,
            'vm_mem_util': vm.memory_utilization,
            'vm_io_util': vm.io_utilization,
            'vm_net_util': vm.network_utilization,
            # Available capacities
            'vm_cpu_avail': vm.cpu_cap - vm.cpu_alloc,
            'vm_mem_avail': vm.mem_cap - vm.mem_alloc,
            'vm_io_avail': vm.io_cap - vm.io_alloc,
            'vm_net_avail': vm.net_cap - vm.net_alloc,
            # VM Total Capacities
            'vm_cpu_cap': vm.cpu_cap,
            'vm_mem_cap': vm.mem_cap,
            'vm_io_cap': vm.io_cap,
            'vm_net_cap': vm.net_cap,
            # Workload Requirements
            'wl_cpu_req': wl.cpu_req,
            'wl_mem_req': wl.mem_req,
            'wl_io_req': wl.io_req,
            'wl_net_req': wl.net_req,
            'wl_type_enc': self.WORKLOAD_ENCODING.get(wl.workload_type, 0)
        }

    def generate_dataset(self, num_vms: int = 20, num_workloads: int = 500, out_dir: str = "data") -> str:
        """
        Generates simulated environments, processes workloads, maps features, calculates 
        ground-truth targets, and saves the CSV without tracking it in Git.
        """
        os.makedirs(out_dir, exist_ok=True)
        env = SimulationEnvironment()
        env.generate_default_vms(num_vms, cpu=1000, mem=16, io=5000, net=1000)
        
        generator = WorkloadGenerator(seed=self.seed)
        workloads = generator.generate_workloads(num_workloads)

        dataset_records = []
        
        for wl in workloads:
            # 1. Capture candidate records for the incoming workload against ALL VMs
            for vm in env.vms.values():
                features = self.extract_features(vm, wl)
                target = self.calculate_ground_truth(vm, wl)
                
                # Combine dictionaries (Target goes at the end)
                record = {**features, 'scenario_id': wl.workload_id, 'target_suitability': target}
                dataset_records.append(record)
            
            # 2. To keep the simulation dynamic, actually allocate the workload 
            # to the BEST feasible VM so the environment evolves.
            feasible_vms = env.get_feasible_vms(wl)
            if feasible_vms:
                # Use our exact ground truth formula to pick the "best" VM in the simulation loop
                # so the state evolves identically to a perfect oracle.
                best_vm = max(feasible_vms, key=lambda v: self.calculate_ground_truth(v, wl))
                env.allocate_workload(best_vm.vm_id, wl)
                
            # Note: We omit releasing workloads in this simple generation loop 
            # to continuously build up pressure and create dense utilization examples.
            # In a full simulation, workloads expire based on their duration.

        df = pd.DataFrame(dataset_records)
        
        # Validation checks
        self.validate_dataset(df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_seed{self.seed}_vms{num_vms}_wl{num_workloads}_{timestamp}.csv"
        filepath = os.path.join(out_dir, filename)
        
        # Save dataset
        df.to_csv(filepath, index=False)
        
        # Save metadata mapping
        meta_filepath = os.path.join(out_dir, f"meta_seed{self.seed}_{timestamp}.json")
        metadata = {
            "seed": self.seed,
            "num_vms": num_vms,
            "num_workloads": num_workloads,
            "columns": list(df.columns),
            "rows": len(df),
            "target": "target_suitability",
            "features_count": len(features)
        }
        with open(meta_filepath, "w") as mf:
            json.dump(metadata, mf, indent=4)
            
        logger.info(f"Dataset generated at {filepath} with {len(df)} records.")
        return filepath

    def validate_dataset(self, df: pd.DataFrame):
        """Validates the generated dataset against basic rules and forbidden leakage."""
        # 1. No Missing Values
        if df.isnull().values.any():
            raise ValueError("Dataset contains missing values.")
            
        # 2. Target Range
        if df['target_suitability'].min() < 0.0 or df['target_suitability'].max() > 100.0:
            raise ValueError("Target suitability out of bounds [0, 100].")
            
        # 3. No leakage variables (anything recording state POST allocation)
        leakage_keywords = ['resulting', 'post', 'future', 'allocated_']
        for col in df.columns:
            for kw in leakage_keywords:
                if kw in col.lower():
                    raise ValueError(f"Potential target leakage detected in feature: {col}")
