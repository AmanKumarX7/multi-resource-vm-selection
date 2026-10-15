import logging
from .connection import get_db_connection

logger = logging.getLogger(__name__)

class ExperimentRepository:
    """Repository handling insertions for Experiment tracking."""
    
    @staticmethod
    def create_experiment(name: str, description: str = "") -> int:
        """Creates an experiment and returns its auto-incremented ID."""
        query = "INSERT INTO experiments (name, description) VALUES (%s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (name, description))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
            
    @staticmethod
    def get_all_experiments():
        """Returns all experiments from the DB."""
        query = "SELECT experiment_id, name, created_at FROM experiments ORDER BY experiment_id DESC"
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            raise
            
    @staticmethod
    def get_experiment_results(experiment_id: int):
        """Returns all results for a given experiment ID."""
        query = "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id = %s"
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (experiment_id,))
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            raise

class VMRepository:
    """Repository handling insertions for VMs."""
    
    @staticmethod
    def insert_vms(experiment_id: int, vms: list) -> dict:
        """Inserts VMs and returns a mapping of vm_id -> internal_id."""
        query = "INSERT INTO vms (experiment_id, vm_id, cpu_cap, mem_cap, io_cap, net_cap) VALUES (%s, %s, %s, %s, %s, %s)"
        mapping = {}
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for vm in vms:
                        cursor.execute(query, (experiment_id, vm.vm_id, vm.cpu_cap, vm.mem_cap, vm.io_cap, vm.net_cap))
                        mapping[vm.vm_id] = cursor.lastrowid
                    conn.commit()
            return mapping
        except Exception as e:
            logger.error(f"Failed to insert VMs: {e}")
            raise

class WorkloadRepository:
    """Repository handling insertions for Workloads."""
    
    @staticmethod
    def insert_workloads(experiment_id: int, workloads: list) -> dict:
        """Inserts Workloads and returns a mapping of workload_id -> internal_id."""
        query = """INSERT INTO workloads (experiment_id, workload_id, workload_type, 
                   cpu_req, mem_req, io_req, net_req, arrival_time, duration) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        mapping = {}
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for wl in workloads:
                        cursor.execute(query, (experiment_id, wl.workload_id, wl.workload_type, 
                                               wl.cpu_req, wl.mem_req, wl.io_req, wl.net_req, 
                                               wl.arrival_time, wl.duration))
                        mapping[wl.workload_id] = cursor.lastrowid
                    conn.commit()
            return mapping
        except Exception as e:
            logger.error(f"Failed to insert workloads: {e}")
            raise

class ModelRepository:
    """Repository handling insertions for ML model tracking."""
    
    @staticmethod
    def create_model_run(experiment_id: int, model_name: str, hyperparameters: str) -> int:
        """Creates a model run record and returns its auto-incremented ID."""
        query = "INSERT INTO model_runs (experiment_id, model_name, hyperparameters) VALUES (%s, %s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (experiment_id, model_name, hyperparameters))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to create model run: {e}")
            raise

    @staticmethod
    def log_metrics(run_id: int, metrics: dict, step: int = None):
        """Logs multiple metrics for a specific model run."""
        query = "INSERT INTO model_metrics (run_id, metric_name, metric_value, step) VALUES (%s, %s, %s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for name, value in metrics.items():
                        cursor.execute(query, (run_id, name, float(value), step))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise

class AllocationRepository:
    """Repository handling insertions for VM allocation decisions."""
    
    @staticmethod
    def log_decision(experiment_id: int, workload_internal_id: int, 
                     vm_internal_id: int, method: str, decision_time: float, 
                     is_successful: bool) -> int:
        """Logs an individual allocation decision."""
        query = """
            INSERT INTO allocation_decisions 
            (experiment_id, workload_internal_id, vm_internal_id, method, decision_time_seconds, is_successful) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Treat vm_internal_id as NULL if it's None or < 0
                    v_id = vm_internal_id if (vm_internal_id is not None and vm_internal_id >= 0) else None
                    cursor.execute(query, (experiment_id, workload_internal_id, v_id, method, decision_time, is_successful))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log allocation decision: {e}")
            raise
