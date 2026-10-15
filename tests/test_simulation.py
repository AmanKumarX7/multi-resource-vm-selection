import pytest
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.simulation.workload_generator import WorkloadGenerator
from src.simulation.environment import SimulationEnvironment

class TestSimulationLayer:

    def test_vm_creation_and_bounds(self):
        vm = VM("VM_Test", 100, 16, 1000, 1000)
        assert vm.cpu_utilization == 0.0
        assert vm.memory_utilization == 0.0
        
        # Fill completely
        res = vm.allocate(100, 16, 1000, 1000)
        assert res is True
        assert vm.cpu_utilization == 100.0
        
        # Try to over-allocate
        assert not vm.can_allocate(1, 1, 1, 1)
        res_fail = vm.allocate(1, 1, 1, 1)
        assert res_fail is False
        assert vm.cpu_utilization == 100.0  # Does not over-100 if allocation failed correctly

        # Release and go below theoretically 0?
        vm.release(100, 16, 1000, 1000)
        assert vm.cpu_utilization == 0.0
        vm.release(50, 50, 50, 50) # Extra release testing safety bounds
        assert vm.cpu_utilization == 0.0
        assert vm.cpu_alloc == 0.0

    def test_feasibility_check(self):
        vm = VM("VM_Feasible", 100, 16, 1000, 1000)
        
        # Test just enough cpu
        assert vm.can_allocate(100, 10, 500, 500)
        assert not vm.can_allocate(101, 10, 500, 500)
        
        # Make inactive
        vm.status = "Maintenance"
        assert not vm.can_allocate(10, 1, 50, 50)

    def test_workload_generator_reproducibility(self):
        gen1 = WorkloadGenerator(seed=42)
        wl1 = gen1.generate_workloads(5)
        
        gen2 = WorkloadGenerator(seed=42)
        wl2 = gen2.generate_workloads(5)
        
        gen3 = WorkloadGenerator(seed=99)
        wl3 = gen3.generate_workloads(5)

        # Checking ID and values match between gen1 and gen2, but diff from gen3
        for w1, w2 in zip(wl1, wl2):
            assert w1.workload_id == w2.workload_id
            assert w1.workload_type == w2.workload_type
            assert w1.cpu_req == w2.cpu_req
            
        assert wl1[0].cpu_req != wl3[0].cpu_req or wl1[0].workload_type != wl3[0].workload_type

    def test_environment_allocation_flow(self):
        env = SimulationEnvironment()
        env.generate_default_vms(2, cpu=500, mem=8, io=1000, net=1000)

        wl = Workload("W_1", "cpu-intensive", 450, 2, 100, 100, 0, 50)
        
        feasible_vms = env.get_feasible_vms(wl)
        assert len(feasible_vms) == 2  # both can fit
        
        # Allocate to VM_0
        res = env.allocate_workload("VM_0", wl)
        assert res is True
        assert env.vms["VM_0"].active_workloads == 1
        assert "W_1" in env.allocations
        
        # Check utilization bounds change
        assert env.vms["VM_0"].cpu_utilization == 90.0 # 450 / 500
        
        # Now VM_0 should not be feasible for a second such workload
        feasible_vms = env.get_feasible_vms(wl)
        assert len(feasible_vms) == 1
        assert feasible_vms[0].vm_id == "VM_1"
        
        # Release workload
        res = env.release_workload("W_1")
        assert res is True
        assert env.vms["VM_0"].active_workloads == 0
        assert env.vms["VM_0"].cpu_utilization == 0.0
