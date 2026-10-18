import os
import json
import logging
import statistics
import pandas as pd
from typing import List, Dict, Any

from src.evaluation.experiment_engine import ExperimentEngine
from src.database.connection import get_db_connection

logger = logging.getLogger(__name__)

class ScalabilityRunner:
    def __init__(self, ml_model_path: str):
        self.engine = ExperimentEngine(ml_model_path)
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'experiments'))
        os.makedirs(self.output_dir, exist_ok=True)
        
    def aggregate_results(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple run results."""
        if not runs:
            return {}
            
        aggregated = {}
        # Keys to aggregate
        numeric_keys = [
            'allocation_successes', 'allocation_failures', 'overloaded_vms',
            'cpu_balance_std', 'mem_balance_std', 'io_balance_std', 'net_balance_std', 'composite_balance_std',
            'mean_decision_time', 'median_decision_time', 'min_decision_time', 'max_decision_time', 'std_decision_time',
            'total_sim_time', 'mean_relative_quality'
        ]
        
        for k in numeric_keys:
            values = [r[k] for r in runs if k in r]
            if not values:
                continue
            aggregated[f"{k}_mean"] = statistics.mean(values)
            aggregated[f"{k}_median"] = statistics.median(values)
            aggregated[f"{k}_min"] = min(values)
            aggregated[f"{k}_max"] = max(values)
            aggregated[f"{k}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            
        return aggregated

    def _persist_aggregate_db(self, name: str, vms: int, wls: int, base_agg: dict, ml_agg: dict):
        """Persist aggregated results back to MySQL as a specialized run."""
        query_exp = "INSERT INTO experiments (name, description) VALUES (%s, %s)"
        query_res = "INSERT INTO experiment_results (experiment_id, metric_name, metric_value) VALUES (%s, %s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Create a marker experiment
                    cursor.execute(query_exp, (f"AGG_{name}_vms{vms}_wls{wls}", "Aggregated Scalability Result"))
                    exp_id = cursor.lastrowid
                    
                    # Store VM and workload counts to make filtering easy for UI
                    cursor.execute(query_res, (exp_id, "config_vms", float(vms)))
                    cursor.execute(query_res, (exp_id, "config_workloads", float(wls)))
                    
                    for k, v in base_agg.items():
                        cursor.execute(query_res, (exp_id, f"baseline_{k}", float(v)))
                        
                    for k, v in ml_agg.items():
                        cursor.execute(query_res, (exp_id, f"random_forest_{k}", float(v)))
                        
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist aggregate to DB: {e}")
            raise

    def run_matrix(self, name: str, vms_list: List[int], workloads_list: List[int], seeds: List[int]):
        """Runs the scalability matrix and saves outputs."""
        logger.info(f"Starting scalability matrix: {name}")
        
        all_results = []
        
        for vms in vms_list:
            for wls in workloads_list:
                logger.info(f"Running config - VMs: {vms}, Workloads: {wls}")
                base_runs = []
                ml_runs = []
                
                for seed in seeds:
                    base_res, ml_res = self.engine.run_comparison(vms, wls, seed)
                    base_runs.append(base_res)
                    ml_runs.append(ml_res)
                    
                    all_results.append({
                        "series": name,
                        "vms": vms,
                        "workloads": wls,
                        "seed": seed,
                        "method": "baseline",
                        **base_res
                    })
                    all_results.append({
                        "series": name,
                        "vms": vms,
                        "workloads": wls,
                        "seed": seed,
                        "method": "random_forest",
                        **ml_res
                    })
                    
                # Aggregate and persist to DB
                base_agg = self.aggregate_results(base_runs)
                ml_agg = self.aggregate_results(ml_runs)
                self._persist_aggregate_db(name, vms, wls, base_agg, ml_agg)
                
        # Save CSV/JSON globally for the matrix
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.output_dir, f"{name}_raw.csv"), index=False)
        with open(os.path.join(self.output_dir, f"{name}_raw.json"), "w") as f:
            json.dump(all_results, f, indent=4)
            
        logger.info(f"Scalability matrix {name} completed.")
