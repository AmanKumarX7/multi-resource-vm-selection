"""
Live State and Lifecycle Management for the Live Interactive Simulator.

Provides state isolation, database persistence synchronization, fair algorithm
comparison execution, and simulation tick lifecycle management.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional

from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.simulation.environment import SimulationEnvironment
from src.algorithms.baseline import BaselineSelector
from src.algorithms.ml_selector import MLSelector
from src.database.repositories import (
    ExperimentRepository,
    VMRepository,
    WorkloadRepository,
    AllocationRepository,
    SimulationEventRepository,
    SystemRepository
)

logger = logging.getLogger(__name__)

DEFAULT_SEED_VMS = [
    {"vm_id": "Node_1", "cpu_cap": 2000.0, "mem_cap": 16.0, "io_cap": 5000.0, "net_cap": 1000.0},
    {"vm_id": "Node_2", "cpu_cap": 4000.0, "mem_cap": 32.0, "io_cap": 8000.0, "net_cap": 2000.0},
    {"vm_id": "Node_3", "cpu_cap": 1500.0, "mem_cap": 8.0, "io_cap": 3000.0, "net_cap": 1000.0},
    {"vm_id": "Node_4", "cpu_cap": 3000.0, "mem_cap": 24.0, "io_cap": 6000.0, "net_cap": 1500.0},
]

DEFAULT_SEED_WORKLOADS = [
    {"workload_id": "WL_Web_1", "workload_type": "web_server", "cpu_req": 150.0, "mem_req": 2.0, "io_req": 200.0, "net_req": 400.0, "duration": 8},
    {"workload_id": "WL_Web_2", "workload_type": "web_server", "cpu_req": 250.0, "mem_req": 4.0, "io_req": 350.0, "net_req": 600.0, "duration": 12},
    {"workload_id": "WL_DB_1", "workload_type": "database", "cpu_req": 500.0, "mem_req": 8.0, "io_req": 1200.0, "net_req": 300.0, "duration": 15},
    {"workload_id": "WL_DB_2", "workload_type": "database", "cpu_req": 800.0, "mem_req": 12.0, "io_req": 2000.0, "net_req": 500.0, "duration": 20},
    {"workload_id": "WL_Batch_1", "workload_type": "batch_job", "cpu_req": 1000.0, "mem_req": 6.0, "io_req": 800.0, "net_req": 150.0, "duration": 25},
    {"workload_id": "WL_Batch_2", "workload_type": "batch_job", "cpu_req": 1200.0, "mem_req": 8.0, "io_req": 1500.0, "net_req": 200.0, "duration": 30},
    {"workload_id": "WL_Cache_1", "workload_type": "cache", "cpu_req": 200.0, "mem_req": 10.0, "io_req": 400.0, "net_req": 800.0, "duration": 10},
    {"workload_id": "WL_Cache_2", "workload_type": "cache", "cpu_req": 300.0, "mem_req": 14.0, "io_req": 600.0, "net_req": 1200.0, "duration": 15},
    {"workload_id": "WL_Stream_1", "workload_type": "balanced", "cpu_req": 350.0, "mem_req": 4.0, "io_req": 300.0, "net_req": 900.0, "duration": 14},
    {"workload_id": "WL_Stream_2", "workload_type": "balanced", "cpu_req": 450.0, "mem_req": 6.0, "io_req": 500.0, "net_req": 1000.0, "duration": 18},
]


def load_ml_selector() -> Tuple[Optional[MLSelector], Optional[str]]:
    """Discovers and instantiates the latest pre-trained Random Forest MLSelector."""
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    if not os.path.exists(models_dir):
        return None, "Models directory not found"
        
    joblibs = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not joblibs:
        return None, "No trained .joblib models found in models/"
        
    latest_file = sorted(joblibs)[-1]
    latest_path = os.path.join(models_dir, latest_file)
    try:
        selector = MLSelector(model_path=latest_path)
        return selector, None
    except Exception as e:
        logger.error(f"Failed to load MLSelector from {latest_path}: {e}")
        return None, str(e)


def initialize_persistent_state() -> dict:
    """
    Connects to MySQL, retrieves or creates the persistent Live Simulator experiment,
    and loads any existing VMs and Workloads.
    """
    from src.database.connection import get_db_connection
    is_connected = False
    try:
        with get_db_connection() as conn:
            is_connected = conn.is_connected()
    except Exception as e:
        logger.warning(f"MySQL unavailable for persistent live state: {e}")
        is_connected = False
    
    live_exp_id = None
    vms_dict: Dict[str, VM] = {}
    workloads_dict: Dict[str, Workload] = {}
    
    if is_connected:
        try:
            live_exp_id = ExperimentRepository.get_or_create_live_experiment()
            
            # Fetch existing VMs from MySQL
            db_vms = VMRepository.get_vms(experiment_id=live_exp_id)
            for row in db_vms:
                if row["vm_id"].startswith("TestNode"):
                    continue
                vm = VM(row["vm_id"], float(row["cpu_cap"]), float(row["mem_cap"]), 
                        float(row["io_cap"]), float(row["net_cap"]))
                vms_dict[vm.vm_id] = vm

            # Fetch existing Workloads from MySQL
            db_wls = WorkloadRepository.get_workloads(experiment_id=live_exp_id)
            for row in db_wls:
                wl = Workload(row["workload_id"], row["workload_type"], float(row["cpu_req"]), 
                              float(row["mem_req"]), float(row["io_req"]), float(row["net_req"]), 
                              int(row.get("arrival_time", 0)), int(row["duration"]))
                workloads_dict[wl.workload_id] = wl

        except Exception as e:
            logger.error(f"Error initializing persistent state from MySQL: {e}")
            is_connected = False

    # Build initial isolated simulation environments
    env_baseline = SimulationEnvironment()
    env_rf = SimulationEnvironment()
    for vm in vms_dict.values():
        env_baseline.add_vm(vm.clone())
        env_rf.add_vm(vm.clone())
        
    return {
        "db_connected": is_connected,
        "live_exp_id": live_exp_id,
        "saved_vms": vms_dict,
        "saved_workloads": workloads_dict,
        "env_baseline": env_baseline,
        "env_rf": env_rf,
    }


