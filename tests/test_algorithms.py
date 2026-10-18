import pytest
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.algorithms.baseline import BaselineSelector

class TestBaselineSelector:

    def test_baseline_infeasible_exclusion(self):
        selector = BaselineSelector()
        wl = Workload("w1", "balanced", 50, 5, 100, 100, 0, 10)
        
        # Test providing empty feasible VMs list
        vm, dt = selector.select_vm([], wl)
        assert vm is None
        assert dt > 0

    def test_baseline_correct_selection(self):
        # VM_1 is heavily loaded, VM_2 is completely free.
        vm1 = VM("VM_1", 100, 16, 1000, 1000)
        vm1.allocate(90, 10, 800, 800)
        vm2 = VM("VM_2", 100, 16, 1000, 1000)
        
        wl = Workload("w1", "balanced", 5, 1, 50, 50, 0, 10)
        feasible = [vm for vm in [vm1, vm2] if vm.can_allocate(wl.cpu_req, wl.mem_req, wl.io_req, wl.net_req)]
        
        assert len(feasible) == 2 
        
        selector = BaselineSelector()
        selected, dt = selector.select_vm(feasible, wl)
        
        assert selected is not None
        assert selected.vm_id == "VM_2" # Prefer less loaded VM
        assert dt > 0

    def test_baseline_weights_affect_selection(self):
        vm1 = VM("VM_1", 100, 16, 1000, 1000)
        vm1.allocate(90, 2, 100, 100) # Highly loaded CPU, freed memory
        
        vm2 = VM("VM_2", 100, 16, 1000, 1000)
        vm2.allocate(10, 14, 100, 100) # Highly loaded memory, freed CPU
        
        wl = Workload("w1", "balanced", 1, 1, 1, 1, 0, 10)
        feasible = [vm1, vm2]
        
        # When caring strictly about CPU (weight completely skewed)
        sel_cpu = BaselineSelector(w_cpu=1.0, w_mem=0.0, w_io=0.0, w_net=0.0)
        chosen1, _ = sel_cpu.select_vm(feasible, wl)
        # VM_2 has lower CPU utilization (10% vs 90%), so it should be chosen
        assert chosen1.vm_id == "VM_2"
        
        # When caring strictly about Memory
        sel_mem = BaselineSelector(w_cpu=0.0, w_mem=1.0, w_io=0.0, w_net=0.0)
        chosen2, _ = sel_mem.select_vm(feasible, wl)
        # VM_1 has lower Memory utilization (12.5% vs 87.5%), so it should be chosen
        assert chosen2.vm_id == "VM_1"

    def test_baseline_deterministic_tie_breaking(self):
        # 3 identical empty VMs
        vm_c = VM("VM_C", 100, 16, 1000, 1000)
        vm_a = VM("VM_A", 100, 16, 1000, 1000)
        vm_b = VM("VM_B", 100, 16, 1000, 1000)
        
        wl = Workload("w1", "balanced", 10, 2, 50, 50, 0, 10)
        feasible = [vm_b, vm_c, vm_a] # Mis-ordered
        
        selector = BaselineSelector()
        selected, dt = selector.select_vm(feasible, wl)
        
        # The scores are identical, so lexical fallback prioritizes "VM_A"
        assert selected.vm_id == "VM_A"
        assert dt > 0

    def test_decision_time_isolated(self):
        vm1 = VM("VM_1", 100, 16, 1000, 1000)
        wl = Workload("w1", "balanced", 10, 2, 50, 50, 0, 10)
        selector = BaselineSelector()
        
        # Execution time measurement
        selected, dt = selector.select_vm([vm1], wl)
        assert isinstance(dt, float)
        assert dt > 0.0
        # Time shouldn't be huge for a basic arithmetic task
        assert dt < 0.1 
