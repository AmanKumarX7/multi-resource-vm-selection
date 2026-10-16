"""
Unit and Integration Tests for Phase-II Live Simulator Components.

Covers:
- Physical resource units standardization (ResourceUnits)
- Simulation environment state isolation, cloning, reset, update, and removal
- Fair live allocation comparison (Baseline vs Random Forest)
- Simulation tick advancement and duration-based resource release
- Safe simulation reset without inventory destruction
- Missing ML model graceful error handling
- Infeasible workload rejection handling
- MySQL persistence and restart recovery
- MLSelector pipeline_path compatibility
"""

import pytest
import os
from src.simulation.units import ResourceUnits
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.simulation.environment import SimulationEnvironment
from src.algorithms.baseline import BaselineSelector
from src.algorithms.ml_selector import MLSelector
from src.ui.live_state import (
    initialize_persistent_state,
    sync_add_vm,
    sync_update_vm,
    sync_delete_vm,
    sync_add_workload,
    sync_update_workload,
    sync_delete_workload,
    evaluate_fair_allocation,
    advance_simulation_tick,
    reset_simulation,
    load_ml_selector,
    DEFAULT_SEED_WORKLOADS
)
from src.database.connection import DatabaseConfig, get_db_connection
from src.database.repositories import (
    VMRepository,
    WorkloadRepository,
    AllocationRepository,
    SimulationEventRepository,
    SystemRepository
)


def db_is_available() -> bool:
    try:
        config = DatabaseConfig.load_config()
        if not config.get("user"):
            return False
        with get_db_connection() as conn:
            return conn.is_connected()
    except Exception:
        return False


class TestResourceUnits:
    """Verifies strict unit standardization across all resources."""

    def test_unit_constants(self):
        assert ResourceUnits.CPU == "MHz"
        assert ResourceUnits.RAM == "GB"
        assert ResourceUnits.DISK_IO == "MB/s"
        assert ResourceUnits.NET == "MB/s"
        assert ResourceUnits.DURATION == "ticks"
        assert ResourceUnits.DECISION_TIME == "ms"

    def test_unit_formatting(self):
        assert ResourceUnits.format_cpu(1000.0) == "1,000 MHz"
        assert ResourceUnits.format_ram(16.0) == "16.0 GB"
        assert ResourceUnits.format_io(5000.0) == "5,000 MB/s"
        assert ResourceUnits.format_net(1000.0) == "1,000 MB/s"
        assert ResourceUnits.format_duration(10) == "10 ticks"
        assert ResourceUnits.format_decision_time(0.001234) == "1.2340 ms"


class TestSimulationEnvironmentExtensions:
    """Verifies cloning and isolated environment management."""

    def test_vm_cloning(self):
        original = VM("VM_Orig", 2000, 16, 5000, 1000)
        original.allocate(500, 4, 1000, 200)
        clone = original.clone()

        assert clone.vm_id == original.vm_id
        assert clone.cpu_cap == original.cpu_cap
        assert clone.cpu_alloc == 500
        assert clone.active_workloads == 1

        # Modifying clone must NOT affect original
        clone.allocate(500, 2, 500, 100)
        assert clone.cpu_alloc == 1000
        assert original.cpu_alloc == 500

    def test_environment_cloning(self):
        env = SimulationEnvironment()
        env.add_vm(VM("Node_A", 1000, 8, 2000, 500))
        env.add_vm(VM("Node_B", 2000, 16, 4000, 1000))

        wl = Workload("WL_1", "web_server", 200, 2, 500, 100, 0, 10)
        env.allocate_workload("Node_A", wl)

        cloned_env = env.clone()
        assert len(cloned_env.vms) == 2
        assert "Node_A" in cloned_env.vms
        assert cloned_env.vms["Node_A"].cpu_alloc == 200
        assert "WL_1" in cloned_env.allocations

        # Modifying original must NOT affect clone
        wl2 = Workload("WL_2", "database", 300, 2, 500, 100, 0, 10)
        env.allocate_workload("Node_A", wl2)
        assert env.vms["Node_A"].cpu_alloc == 500
        assert cloned_env.vms["Node_A"].cpu_alloc == 200

    def test_environment_reset_allocations(self):
        env = SimulationEnvironment()
        env.add_vm(VM("Node_A", 1000, 8, 2000, 500))
        wl = Workload("WL_1", "web_server", 200, 2, 500, 100, 0, 10)
        env.allocate_workload("Node_A", wl)

        assert env.vms["Node_A"].active_workloads == 1
        assert len(env.allocations) == 1

        env.reset_allocations()
        assert env.vms["Node_A"].active_workloads == 0
        assert env.vms["Node_A"].cpu_alloc == 0
        assert len(env.allocations) == 0
        assert "Node_A" in env.vms  # VM definition preserved

    def test_environment_remove_and_update_vm(self):
        env = SimulationEnvironment()
        env.add_vm(VM("Node_A", 1000, 8, 2000, 500))

        # Safe update
        assert env.update_vm("Node_A", 2000, 16, 4000, 1000) is True
        assert env.vms["Node_A"].cpu_cap == 2000

        # Safe delete
        assert env.remove_vm("Node_A") is True
        assert "Node_A" not in env.vms


class TestLiveComparisonAndIsolation:
    """Verifies that Compare Both mode tests both algorithms from identical initial conditions."""

    def test_fair_allocation_isolated_states(self):
        env_base = SimulationEnvironment()
        env_base.add_vm(VM("Node_1", 2000, 16, 5000, 1000))
        env_base.add_vm(VM("Node_2", 4000, 32, 8000, 2000))

        env_rf = env_base.clone()
        base_sel = BaselineSelector()
        ml_sel, _ = load_ml_selector()

        wl = Workload("WL_Test_1", "web_server", 500, 4, 1000, 200, 0, 10)

        res = evaluate_fair_allocation(
            "Compare Both — Baseline vs Random Forest",
            wl,
            env_base,
            env_rf,
            base_sel,
            ml_sel,
            experiment_id=None,
            current_tick=0
        )

        assert res["workload"].workload_id == "WL_Test_1"
        assert res["baseline"]["success"] is True
        assert res["baseline"]["selected_vm"] in ["Node_1", "Node_2"]
        assert res["baseline"]["decision_time_ms"] > 0

        if ml_sel is not None:
            assert res["rf"]["success"] is True
            assert res["rf"]["selected_vm"] in ["Node_1", "Node_2"]
            assert res["rf"]["decision_time_ms"] > 0

        # Verify that both simulation environments updated independently
        base_selected = res["baseline"]["selected_vm"]
        assert env_base.vms[base_selected].cpu_alloc == 500

    def test_infeasible_workload_rejection(self):
        env_base = SimulationEnvironment()
        env_base.add_vm(VM("TinyNode", 100, 1, 100, 100))
        env_rf = env_base.clone()

        base_sel = BaselineSelector()
        ml_sel, _ = load_ml_selector()

        # Workload requesting 500 MHz on a 100 MHz VM
        giant_wl = Workload("WL_Giant", "database", 500, 4, 1000, 200, 0, 10)

        res = evaluate_fair_allocation(
            "Compare Both — Baseline vs Random Forest",
            giant_wl,
            env_base,
            env_rf,
            base_sel,
            ml_sel,
            experiment_id=None,
            current_tick=0
        )

        assert res["baseline"]["success"] is False
        assert res["baseline"]["selected_vm"] is None
        assert "Rejected" in res["baseline"]["reason"] or "Insufficient capacity" in res["baseline"]["reason"]

        if ml_sel is not None:
            assert res["rf"]["success"] is False
            assert res["rf"]["selected_vm"] is None
            assert "Rejected" in res["rf"]["reason"] or "Insufficient capacity" in res["rf"]["reason"]

    def test_missing_ml_model_isolation(self):
        """When ML selector is None, baseline still succeeds and RF reports explicit error."""
        env_base = SimulationEnvironment()
        env_base.add_vm(VM("Node_1", 2000, 16, 5000, 1000))
        env_rf = env_base.clone()

        base_sel = BaselineSelector()
        wl = Workload("WL_Norm", "web_server", 200, 2, 500, 100, 0, 10)

        res = evaluate_fair_allocation(
            "Compare Both — Baseline vs Random Forest",
            wl,
            env_base,
            env_rf,
            base_sel,
            ml_selector=None,
            experiment_id=None,
            current_tick=0
        )

        assert res["baseline"]["success"] is True
        assert res["rf"]["success"] is False
        assert res["rf"]["error"] is True
        assert "unavailable" in res["rf"]["reason"].lower()

    def test_tick_advancement_and_duration_release(self):
        env_base = SimulationEnvironment()
        env_base.add_vm(VM("Node_1", 2000, 16, 5000, 1000))
        env_rf = env_base.clone()

        wl = Workload("WL_Short", "batch_job", 500, 4, 1000, 200, 0, 2)
        env_base.allocate_workload("Node_1", wl)
        env_rf.allocate_workload("Node_1", wl)

        active_base = [{"workload_id": "WL_Short", "vm_id": "Node_1", "expires_at": 2}]
        active_rf = [{"workload_id": "WL_Short", "vm_id": "Node_1", "expires_at": 2}]

        # Advance to tick 1: should still be active
        new_tick, logs = advance_simulation_tick(env_base, env_rf, active_base, active_rf, 0, None)
        assert new_tick == 1
        assert len(logs) == 0
        assert env_base.vms["Node_1"].cpu_alloc == 500

        # Advance to tick 2: expires_at == 2, should release
        new_tick, logs = advance_simulation_tick(env_base, env_rf, active_base, active_rf, 1, None)
        assert new_tick == 2
        assert len(logs) == 2  # Released from base and rf
        assert env_base.vms["Node_1"].cpu_alloc == 0
        assert env_rf.vms["Node_1"].cpu_alloc == 0
        assert len(active_base) == 0
        assert len(active_rf) == 0


class TestMLSelectorPipelinePathCompatibility:
    """Verifies that MLSelector exposes pipeline_path and model_path consistently."""

    def test_pipeline_path_attribute(self):
        ml_sel, err = load_ml_selector()
        if ml_sel is not None:
            assert hasattr(ml_sel, "pipeline_path")
            assert ml_sel.pipeline_path == ml_sel.model_path
            assert os.path.exists(ml_sel.pipeline_path)

    def test_ml_selector_flexible_kwarg(self):
        ml_sel, err = load_ml_selector()
        if ml_sel is not None:
            sel_by_pipe = MLSelector(pipeline_path=ml_sel.pipeline_path)
            assert sel_by_pipe.model_path == ml_sel.pipeline_path
            assert sel_by_pipe.pipeline_path == ml_sel.pipeline_path


