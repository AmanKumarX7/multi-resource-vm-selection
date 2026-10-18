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


def seed_standard_workloads(saved_workloads: Dict[str, Workload], exp_id: Optional[int]) -> int:
    """Helper to seed at least 10 realistic workload definitions into MySQL."""
    added = 0
    for item in DEFAULT_SEED_WORKLOADS:
        wl = Workload(item["workload_id"], item["workload_type"], item["cpu_req"], 
                      item["mem_req"], item["io_req"], item["net_req"], 0, item["duration"])
        ok, _ = sync_add_workload(wl, saved_workloads, exp_id)
        if ok:
            added += 1
    return added


def sync_add_vm(vm: VM, env_baseline: SimulationEnvironment, env_rf: SimulationEnvironment, 
                saved_vms: Dict[str, VM], exp_id: Optional[int]) -> Tuple[bool, str]:
    """Adds a VM definition to MySQL and both isolated simulation environments."""
    if vm.vm_id in env_baseline.vms:
        if vm.vm_id.startswith("TestNode"):
            env_baseline.remove_vm(vm.vm_id)
            env_rf.remove_vm(vm.vm_id)
            saved_vms.pop(vm.vm_id, None)
        else:
            return False, f"VM '{vm.vm_id}' already exists in live pool."
        
    if exp_id:
        try:
            existing_id = VMRepository.get_vm_internal_id(vm.vm_id, exp_id)
            if existing_id:
                VMRepository.update_vm_capacity(vm.vm_id, vm.cpu_cap, vm.mem_cap, vm.io_cap, vm.net_cap, experiment_id=exp_id)
            else:
                VMRepository.insert_single_vm(exp_id, vm)
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="VM_CREATED",
                algorithm="SYSTEM",
                vm_id=vm.vm_id,
                result="SUCCESS",
                message=f"Provisioned VM {vm.vm_id} ({vm.cpu_cap} MHz, {vm.mem_cap} GB, {vm.io_cap} MB/s, {vm.net_cap} MB/s)"
            )
        except Exception as e:
            logger.error(f"Failed to persist VM {vm.vm_id}: {e}")
            return False, f"Database persistence failed: {e}"
            
    saved_vms[vm.vm_id] = vm
    env_baseline.add_vm(vm.clone())
    env_rf.add_vm(vm.clone())
    return True, f"Successfully provisioned VM '{vm.vm_id}'."


def sync_update_vm(vm_id: str, cpu_cap: float, mem_cap: float, io_cap: float, net_cap: float,
                   env_baseline: SimulationEnvironment, env_rf: SimulationEnvironment,
                   saved_vms: Dict[str, VM], exp_id: Optional[int]) -> Tuple[bool, str]:
    """Safely updates VM capacities in MySQL and both environments."""
    if vm_id not in saved_vms:
        return False, f"VM '{vm_id}' not found."
        
    # Check if capacity reduction would violate current allocations in either env
    base_vm = env_baseline.vms.get(vm_id)
    rf_vm = env_rf.vms.get(vm_id)
    
    max_cpu_alloc = max(base_vm.cpu_alloc if base_vm else 0, rf_vm.cpu_alloc if rf_vm else 0)
    max_mem_alloc = max(base_vm.mem_alloc if base_vm else 0, rf_vm.mem_alloc if rf_vm else 0)
    max_io_alloc = max(base_vm.io_alloc if base_vm else 0, rf_vm.io_alloc if rf_vm else 0)
    max_net_alloc = max(base_vm.net_alloc if base_vm else 0, rf_vm.net_alloc if rf_vm else 0)
    
    if (cpu_cap < max_cpu_alloc or mem_cap < max_mem_alloc or 
        io_cap < max_io_alloc or net_cap < max_net_alloc):
        return False, "Cannot reduce capacity below currently allocated resources on active workloads."
        
    if exp_id:
        try:
            VMRepository.update_vm_capacity(vm_id, cpu_cap, mem_cap, io_cap, net_cap, experiment_id=exp_id)
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="VM_UPDATED",
                algorithm="SYSTEM",
                vm_id=vm_id,
                result="SUCCESS",
                message=f"Updated VM {vm_id} capacities to {cpu_cap} MHz, {mem_cap} GB, {io_cap} MB/s, {net_cap} MB/s"
            )
        except Exception as e:
            return False, f"Database update failed: {e}"
            
    saved_vms[vm_id].cpu_cap = cpu_cap
    saved_vms[vm_id].mem_cap = mem_cap
    saved_vms[vm_id].io_cap = io_cap
    saved_vms[vm_id].net_cap = net_cap
    
    env_baseline.update_vm(vm_id, cpu_cap, mem_cap, io_cap, net_cap)
    env_rf.update_vm(vm_id, cpu_cap, mem_cap, io_cap, net_cap)
    return True, f"Successfully updated capacities for VM '{vm_id}'."


def sync_delete_vm(vm_id: str, env_baseline: SimulationEnvironment, env_rf: SimulationEnvironment,
                   saved_vms: Dict[str, VM], exp_id: Optional[int]) -> Tuple[bool, str]:
    """Deletes a VM if it has no active allocations in either environment."""
    if vm_id not in saved_vms:
        return False, f"VM '{vm_id}' not found."
        
    base_vm = env_baseline.vms.get(vm_id)
    rf_vm = env_rf.vms.get(vm_id)
    
    if (base_vm and base_vm.active_workloads > 0) or (rf_vm and rf_vm.active_workloads > 0):
        return False, f"Cannot delete VM '{vm_id}': has active workloads."
        
    if exp_id:
        try:
            VMRepository.delete_vm(vm_id, experiment_id=exp_id)
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="VM_DELETED",
                algorithm="SYSTEM",
                vm_id=vm_id,
                result="SUCCESS",
                message=f"Safely removed VM {vm_id}"
            )
        except Exception as e:
            return False, f"Database delete failed: {e}"
            
    del saved_vms[vm_id]
    env_baseline.remove_vm(vm_id)
    env_rf.remove_vm(vm_id)
    return True, f"Successfully removed VM '{vm_id}'."


def sync_add_workload(wl: Workload, saved_workloads: Dict[str, Workload], 
                      exp_id: Optional[int]) -> Tuple[bool, str]:
    """Saves a workload definition to MySQL and the persistent dictionary."""
    if exp_id:
        try:
            existing_id = WorkloadRepository.get_workload_internal_id(wl.workload_id, exp_id)
            if existing_id:
                WorkloadRepository.update_workload(
                    wl.workload_id, wl.workload_type, wl.cpu_req, wl.mem_req, wl.io_req, wl.net_req, wl.duration, experiment_id=exp_id
                )
            else:
                WorkloadRepository.insert_single_workload(exp_id, wl)
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="WORKLOAD_CREATED",
                algorithm="SYSTEM",
                workload_id=wl.workload_id,
                result="SUCCESS",
                message=f"Created workload {wl.workload_id} ({wl.workload_type})"
            )
        except Exception as e:
            return False, f"Database persistence failed: {e}"
            
    saved_workloads[wl.workload_id] = wl
    return True, f"Saved workload '{wl.workload_id}'."


def sync_update_workload(workload_id: str, workload_type: str, cpu_req: float, mem_req: float,
                         io_req: float, net_req: float, duration: int,
                         saved_workloads: Dict[str, Workload], exp_id: Optional[int]) -> Tuple[bool, str]:
    """Updates an existing workload definition in MySQL and persistent dictionary."""
    if workload_id not in saved_workloads:
        return False, f"Workload '{workload_id}' not found."
        
    if exp_id:
        try:
            WorkloadRepository.update_workload(
                workload_id, workload_type, cpu_req, mem_req, io_req, net_req, duration, experiment_id=exp_id
            )
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="WORKLOAD_UPDATED",
                algorithm="SYSTEM",
                workload_id=workload_id,
                result="SUCCESS",
                message=f"Updated workload {workload_id} requirements"
            )
        except Exception as e:
            return False, f"Database update failed: {e}"
            
    saved_workloads[workload_id] = Workload(
        workload_id, workload_type, cpu_req, mem_req, io_req, net_req, 0, duration
    )
    return True, f"Updated workload '{workload_id}'."


def sync_delete_workload(workload_id: str, saved_workloads: Dict[str, Workload],
                         exp_id: Optional[int]) -> Tuple[bool, str]:
    """Deletes a workload definition from MySQL and dictionary."""
    if workload_id not in saved_workloads:
        return False, f"Workload '{workload_id}' not found."
        
    if exp_id:
        try:
            WorkloadRepository.delete_workload(workload_id, experiment_id=exp_id)
            SimulationEventRepository.log_event(
                experiment_id=exp_id,
                event_type="WORKLOAD_DELETED",
                algorithm="SYSTEM",
                workload_id=workload_id,
                result="SUCCESS",
                message=f"Removed workload definition {workload_id}"
            )
        except Exception as e:
            return False, f"Database delete failed: {e}"
            
    del saved_workloads[workload_id]
    return True, f"Deleted workload '{workload_id}'."


def evaluate_fair_allocation(
    mode: str,
    workload: Workload,
    env_baseline: SimulationEnvironment,
    env_rf: SimulationEnvironment,
    baseline_selector: BaselineSelector,
    ml_selector: Optional[MLSelector] = None,
    exp_id: Optional[int] = None,
    current_tick: int = 0,
    experiment_id: Optional[int] = None
) -> dict:
    """
    Evaluates an incoming workload fairly.
    In 'Compare Both' mode:
    - Baseline evaluates against its isolated environment.
    - Random Forest evaluates against its isolated environment.
    - Both decisions and performance metrics are tracked separately and logged to MySQL.
    """
    effective_exp_id = exp_id if exp_id is not None else experiment_id
    result = {
        "workload": workload,
        "mode": mode,
        "tick": current_tick,
        "baseline": None,
        "rf": None,
        "match_summary": ""
    }
    
    run_baseline = mode.startswith("Compare") or "Baseline" in mode
    run_rf = mode.startswith("Compare") or "Random Forest" in mode

    # 1. Baseline Evaluation
    if run_baseline:
        feasible_base = env_baseline.get_feasible_vms(workload)
        t0 = time.perf_counter_ns()
        selected_base, _ = baseline_selector.select_vm(feasible_base, workload)
        dt_base_sec = (time.perf_counter_ns() - t0) / 1_000_000_000.0
        dt_base_ms = dt_base_sec * 1000.0
        
        success_base = False
        reason_base = ""
        if not feasible_base:
            reason_base = "Rejected: Insufficient capacity on all provisioned VMs."
        elif selected_base:
            success_base = env_baseline.allocate_workload(selected_base.vm_id, workload)
            reason_base = f"Selected least resultant weighted load ({selected_base.vm_id})"
        else:
            reason_base = "Rejected by selector."
            
        result["baseline"] = {
            "selected_vm": selected_base.vm_id if selected_base else None,
            "feasible_count": len(feasible_base),
            "decision_time_ms": dt_base_ms,
            "decision_time_sec": dt_base_sec,
            "success": success_base,
            "reason": reason_base,
            "vm_state_after": selected_base.to_dict() if (selected_base and success_base) else None
        }
        
        # Log to MySQL
        if effective_exp_id:
            try:
                wl_int_id = WorkloadRepository.get_workload_internal_id(workload.workload_id, effective_exp_id)
                vm_int_id = VMRepository.get_vm_internal_id(selected_base.vm_id, effective_exp_id) if selected_base else None
                if wl_int_id:
                    AllocationRepository.log_decision(
                        experiment_id=effective_exp_id,
                        workload_internal_id=wl_int_id,
                        vm_internal_id=vm_int_id,
                        method="baseline",
                        decision_time=dt_base_sec,
                        is_successful=success_base
                    )
                SimulationEventRepository.log_event(
                    experiment_id=effective_exp_id,
                    event_type="ALLOCATION_DECISION",
                    algorithm="BASELINE",
                    workload_id=workload.workload_id,
                    vm_id=selected_base.vm_id if selected_base else None,
                    result="SUCCESS" if success_base else "REJECTED",
                    message=f"Baseline -> {selected_base.vm_id if selected_base else 'None'} in {dt_base_ms:.3f} ms. {reason_base}"
                )
            except Exception as e:
                logger.error(f"Failed to log Baseline decision to MySQL: {e}")

    # 2. Random Forest Evaluation
    if run_rf:
        feasible_rf = env_rf.get_feasible_vms(workload)
        if ml_selector is None:
            result["rf"] = {
                "selected_vm": None,
                "feasible_count": len(feasible_rf),
                "decision_time_ms": 0.0,
                "decision_time_sec": 0.0,
                "success": False,
                "reason": "Error: Random Forest model artifact is unavailable or not loaded.",
                "vm_state_after": None,
                "error": True
            }
        else:
            t0 = time.perf_counter_ns()
            try:
                selected_rf, _ = ml_selector.select_vm(feasible_rf, workload)
                dt_rf_sec = (time.perf_counter_ns() - t0) / 1_000_000_000.0
                dt_rf_ms = dt_rf_sec * 1000.0
                
                success_rf = False
                reason_rf = ""
                if not feasible_rf:
                    reason_rf = "Rejected: Insufficient capacity on all provisioned VMs."
                elif selected_rf:
                    success_rf = env_rf.allocate_workload(selected_rf.vm_id, workload)
                    reason_rf = f"Selected highest predicted multi-resource suitability ({selected_rf.vm_id})"
                else:
                    reason_rf = "Rejected by ML selector."
                    
                result["rf"] = {
                    "selected_vm": selected_rf.vm_id if selected_rf else None,
                    "feasible_count": len(feasible_rf),
                    "decision_time_ms": dt_rf_ms,
                    "decision_time_sec": dt_rf_sec,
                    "success": success_rf,
                    "reason": reason_rf,
                    "vm_state_after": selected_rf.to_dict() if (selected_rf and success_rf) else None,
                    "error": None
                }
                
                # Log to MySQL
                if effective_exp_id:
                    try:
                        wl_int_id = WorkloadRepository.get_workload_internal_id(workload.workload_id, effective_exp_id)
                        vm_int_id = VMRepository.get_vm_internal_id(selected_rf.vm_id, effective_exp_id) if selected_rf else None
                        if wl_int_id:
                            AllocationRepository.log_decision(
                                experiment_id=effective_exp_id,
                                workload_internal_id=wl_int_id,
                                vm_internal_id=vm_int_id,
                                method="random_forest",
                                decision_time=dt_rf_sec,
                                is_successful=success_rf
                            )
                        SimulationEventRepository.log_event(
                            experiment_id=effective_exp_id,
                            event_type="ALLOCATION_DECISION",
                            algorithm="RANDOM_FOREST",
                            workload_id=workload.workload_id,
                            vm_id=selected_rf.vm_id if selected_rf else None,
                            result="SUCCESS" if success_rf else "REJECTED",
                            message=f"Random Forest -> {selected_rf.vm_id if selected_rf else 'None'} in {dt_rf_ms:.3f} ms. {reason_rf}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to log Random Forest decision to MySQL: {e}")
            except Exception as e:
                result["rf"] = {
                    "selected_vm": None,
                    "feasible_count": len(feasible_rf),
                    "decision_time_ms": 0.0,
                    "decision_time_sec": 0.0,
                    "success": False,
                    "reason": f"Random Forest evaluation error: {e}",
                    "vm_state_after": None,
                    "error": str(e)
                }

    # Comparison summary for dual mode
    if result["baseline"] and result["rf"]:
        b_vm = result["baseline"]["selected_vm"]
        r_vm = result["rf"]["selected_vm"]
        if b_vm and r_vm:
            if b_vm == r_vm:
                result["match_summary"] = f"Identical Selection: Both algorithms selected {b_vm}."
            else:
                result["match_summary"] = f"Divergent Selection: Baseline selected {b_vm}, while Random Forest selected {r_vm}."
        elif not b_vm and not r_vm:
            result["match_summary"] = "Both algorithms rejected the workload due to identical feasibility constraints."
        else:
            result["match_summary"] = f"Partial Allocation: Baseline -> {b_vm or 'Rejected'} vs RF -> {r_vm or 'Rejected'}."
            
    return result


def advance_simulation_tick(
    env_baseline: SimulationEnvironment,
    env_rf: SimulationEnvironment,
    active_baseline: List[dict],
    active_rf: List[dict],
    current_tick: int,
    exp_id: Optional[int]
) -> Tuple[int, List[str]]:
    """
    Advances simulation by one tick, checks duration of active workloads,
    and safely releases completed workloads.
    """
    next_tick = current_tick + 1
    events = []
    
    # Check baseline active workloads
    remaining_base = []
    for item in active_baseline:
        if item["expires_at"] <= next_tick:
            env_baseline.release_workload(item["workload_id"])
            msg = f"[Tick {next_tick}] Workload {item['workload_id']} completed on Baseline ({item['vm_id']})."
            events.append(msg)
            if exp_id:
                SimulationEventRepository.log_event(
                    experiment_id=exp_id,
                    event_type="WORKLOAD_RELEASED",
                    algorithm="BASELINE",
                    workload_id=item["workload_id"],
                    vm_id=item["vm_id"],
                    result="COMPLETED",
                    message=msg
                )
        else:
            remaining_base.append(item)
    active_baseline.clear()
    active_baseline.extend(remaining_base)

    # Check Random Forest active workloads
    remaining_rf = []
    for item in active_rf:
        if item["expires_at"] <= next_tick:
            env_rf.release_workload(item["workload_id"])
            msg = f"[Tick {next_tick}] Workload {item['workload_id']} completed on RF ({item['vm_id']})."
            events.append(msg)
            if exp_id:
                SimulationEventRepository.log_event(
                    experiment_id=exp_id,
                    event_type="WORKLOAD_RELEASED",
                    algorithm="RANDOM_FOREST",
                    workload_id=item["workload_id"],
                    vm_id=item["vm_id"],
                    result="COMPLETED",
                    message=msg
                )
        else:
            remaining_rf.append(item)
    active_rf.clear()
    active_rf.extend(remaining_rf)
    
    return next_tick, events


def reset_simulation(
    env_baseline: SimulationEnvironment,
    env_rf: SimulationEnvironment,
    exp_id: Optional[int]
):
    """Resets live allocation states safely without deleting persistent definitions."""
    env_baseline.reset_allocations()
    env_rf.reset_allocations()
    if exp_id:
        SimulationEventRepository.log_event(
            experiment_id=exp_id,
            event_type="SIMULATION_RESET",
            algorithm="SYSTEM",
            result="SUCCESS",
            message="Simulator state reset to 0. VM and workload persistent inventory retained."
        )
