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

    @staticmethod
    def get_or_create_live_experiment() -> int:
        """Retrieves or creates the designated experiment for persistent Live Simulator state."""
        select_query = "SELECT experiment_id FROM experiments WHERE name = 'LIVE_SIMULATOR_STORE' LIMIT 1"
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(select_query)
                    row = cursor.fetchone()
                    if row:
                        return row['experiment_id']
        except Exception as e:
            logger.error(f"Error checking live experiment: {e}")

        # If not found or on fresh DB, create it
        return ExperimentRepository.create_experiment(
            name="LIVE_SIMULATOR_STORE",
            description="Phase-II Live Interactive Simulator Persistent Store"
        )

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

    @staticmethod
    def insert_single_vm(experiment_id: int, vm) -> int:
        """Inserts a single VM definition into MySQL."""
        query = "INSERT INTO vms (experiment_id, vm_id, cpu_cap, mem_cap, io_cap, net_cap) VALUES (%s, %s, %s, %s, %s, %s)"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (experiment_id, vm.vm_id, vm.cpu_cap, vm.mem_cap, vm.io_cap, vm.net_cap))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert single VM: {e}")
            raise

    @staticmethod
    def update_vm_capacity(vm_id: str, cpu_cap: float, mem_cap: float, io_cap: float, net_cap: float, experiment_id: int = None) -> bool:
        """Updates capacities of an existing VM."""
        if experiment_id:
            query = """UPDATE vms SET cpu_cap = %s, mem_cap = %s, io_cap = %s, net_cap = %s 
                       WHERE vm_id = %s AND experiment_id = %s"""
            params = (cpu_cap, mem_cap, io_cap, net_cap, vm_id, experiment_id)
        else:
            query = """UPDATE vms SET cpu_cap = %s, mem_cap = %s, io_cap = %s, net_cap = %s 
                       WHERE vm_id = %s"""
            params = (cpu_cap, mem_cap, io_cap, net_cap, vm_id)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update VM {vm_id}: {e}")
            raise

    @staticmethod
    def delete_vm(vm_id: str, experiment_id: int = None) -> bool:
        """Deletes a VM definition from MySQL."""
        if experiment_id:
            query = "DELETE FROM vms WHERE vm_id = %s AND experiment_id = %s"
            params = (vm_id, experiment_id)
        else:
            query = "DELETE FROM vms WHERE vm_id = %s"
            params = (vm_id,)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete VM {vm_id}: {e}")
            raise

    @staticmethod
    def get_vms(experiment_id: int = None) -> list:
        """Retrieves persistent VM records from MySQL."""
        if experiment_id:
            query = "SELECT internal_id, experiment_id, vm_id, cpu_cap, mem_cap, io_cap, net_cap, created_at FROM vms WHERE experiment_id = %s ORDER BY internal_id ASC"
            params = (experiment_id,)
        else:
            query = "SELECT internal_id, experiment_id, vm_id, cpu_cap, mem_cap, io_cap, net_cap, created_at FROM vms ORDER BY internal_id ASC"
            params = ()
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch VMs: {e}")
            raise

    @staticmethod
    def get_vm_internal_id(vm_id: str, experiment_id: int = None) -> int:
        """Looks up the database internal_id for a given vm_id string."""
        if experiment_id:
            query = "SELECT internal_id FROM vms WHERE vm_id = %s AND experiment_id = %s LIMIT 1"
            params = (vm_id, experiment_id)
        else:
            query = "SELECT internal_id FROM vms WHERE vm_id = %s ORDER BY internal_id DESC LIMIT 1"
            params = (vm_id,)
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    row = cursor.fetchone()
                    return row['internal_id'] if row else None
        except Exception as e:
            logger.error(f"Failed to find VM internal ID: {e}")
            return None

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

    @staticmethod
    def insert_single_workload(experiment_id: int, wl) -> int:
        """Inserts a single Workload definition into MySQL."""
        query = """INSERT INTO workloads (experiment_id, workload_id, workload_type, 
                   cpu_req, mem_req, io_req, net_req, arrival_time, duration) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (experiment_id, wl.workload_id, wl.workload_type, 
                                           wl.cpu_req, wl.mem_req, wl.io_req, wl.net_req, 
                                           wl.arrival_time, wl.duration))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert single workload: {e}")
            raise

    @staticmethod
    def update_workload(workload_id: str, workload_type: str, cpu_req: float, mem_req: float, 
                        io_req: float, net_req: float, duration: int, experiment_id: int = None) -> bool:
        """Updates an existing workload definition."""
        if experiment_id:
            query = """UPDATE workloads SET workload_type = %s, cpu_req = %s, mem_req = %s, 
                       io_req = %s, net_req = %s, duration = %s 
                       WHERE workload_id = %s AND experiment_id = %s"""
            params = (workload_type, cpu_req, mem_req, io_req, net_req, duration, workload_id, experiment_id)
        else:
            query = """UPDATE workloads SET workload_type = %s, cpu_req = %s, mem_req = %s, 
                       io_req = %s, net_req = %s, duration = %s 
                       WHERE workload_id = %s"""
            params = (workload_type, cpu_req, mem_req, io_req, net_req, duration, workload_id)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update workload {workload_id}: {e}")
            raise

    @staticmethod
    def delete_workload(workload_id: str, experiment_id: int = None) -> bool:
        """Deletes a workload definition from MySQL."""
        if experiment_id:
            query = "DELETE FROM workloads WHERE workload_id = %s AND experiment_id = %s"
            params = (workload_id, experiment_id)
        else:
            query = "DELETE FROM workloads WHERE workload_id = %s"
            params = (workload_id,)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete workload {workload_id}: {e}")
            raise

    @staticmethod
    def get_workloads(experiment_id: int = None) -> list:
        """Retrieves persistent workload records from MySQL."""
        if experiment_id:
            query = """SELECT internal_id, experiment_id, workload_id, workload_type, 
                              cpu_req, mem_req, io_req, net_req, arrival_time, duration, created_at 
                       FROM workloads WHERE experiment_id = %s ORDER BY internal_id ASC"""
            params = (experiment_id,)
        else:
            query = """SELECT internal_id, experiment_id, workload_id, workload_type, 
                              cpu_req, mem_req, io_req, net_req, arrival_time, duration, created_at 
                       FROM workloads ORDER BY internal_id ASC"""
            params = ()
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch workloads: {e}")
            raise

    @staticmethod
    def get_workload_internal_id(workload_id: str, experiment_id: int = None) -> int:
        """Looks up the database internal_id for a given workload_id string."""
        if experiment_id:
            query = "SELECT internal_id FROM workloads WHERE workload_id = %s AND experiment_id = %s LIMIT 1"
            params = (workload_id, experiment_id)
        else:
            query = "SELECT internal_id FROM workloads WHERE workload_id = %s ORDER BY internal_id DESC LIMIT 1"
            params = (workload_id,)
        try:
            with get_db_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    row = cursor.fetchone()
                    return row['internal_id'] if row else None
        except Exception as e:
            logger.error(f"Failed to find workload internal ID: {e}")
            return None

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
                    v_id = vm_internal_id if (vm_internal_id is not None and vm_internal_id >= 0) else None
                    cursor.execute(query, (experiment_id, workload_internal_id, v_id, method, decision_time, is_successful))
                    conn.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log allocation decision: {e}")
            raise

