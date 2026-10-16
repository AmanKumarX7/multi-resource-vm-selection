import logging
from typing import List, Dict, Optional
from .vm import VM
from .workload import Workload

logger = logging.getLogger(__name__)

class SimulationEnvironment:
    """
    Manages the overarching simulated environment, tracking a pool of VMs 
    and enabling allocation / deallocation of workloads safely.
    """

    def __init__(self):
        self.vms: Dict[str, VM] = {}
        # Tracks which VM is hosting which Workload
        self.allocations: Dict[str, str] = {} # workload_id -> vm_id
        self.workloads: Dict[str, Workload] = {} # workload_id -> Workload

    def add_vm(self, vm: VM):
        if vm.vm_id in self.vms:
            logger.warning(f"VM {vm.vm_id} already exists in the environment.")
            return
        self.vms[vm.vm_id] = vm

    def generate_default_vms(self, count: int, cpu: float=1000, mem: float=16, io: float=5000, net: float=1000):
        """Helper to create homogeneous VMs for simple environments."""
        for i in range(count):
            vm_id = f"VM_{i}"
            vm = VM(vm_id, cpu, mem, io, net)
            self.add_vm(vm)

    def get_feasible_vms(self, workload: Workload) -> List[VM]:
        """Returns a list of VMs capable of allocating the given workload."""
        return [vm for vm in self.vms.values() 
                if vm.can_allocate(workload.cpu_req, workload.mem_req, workload.io_req, workload.net_req)]

    def allocate_workload(self, vm_id: str, workload: Workload) -> bool:
        """Attempts to allocate a workload to the specified VM."""
        if vm_id not in self.vms:
            logger.error(f"Cannot allocate to nonexistent VM: {vm_id}")
            return False
            
        if workload.workload_id in self.allocations:
            logger.warning(f"Workload {workload.workload_id} is already allocated.")
            return False

        vm = self.vms[vm_id]
        success = vm.allocate(workload.cpu_req, workload.mem_req, workload.io_req, workload.net_req)
        
        if success:
            self.allocations[workload.workload_id] = vm_id
            self.workloads[workload.workload_id] = workload
            
        return success

    def release_workload(self, workload_id: str) -> bool:
        """Releases the specified workload from its assigned VM and updates resources."""
        if workload_id not in self.allocations:
            logger.error(f"Cannot release nonexistent/unallocated workload: {workload_id}")
            return False
            
        vm_id = self.allocations[workload_id]
        workload = self.workloads[workload_id]
        vm = self.vms[vm_id]
        
        # Release the resources safely
        vm.release(workload.cpu_req, workload.mem_req, workload.io_req, workload.net_req)
        
        # Clean up mapping
        del self.allocations[workload_id]
        del self.workloads[workload_id]
        
        return True

    def remove_vm(self, vm_id: str) -> bool:
        """Removes a VM only if it has no active allocations."""
        if vm_id not in self.vms:
            return False
        vm = self.vms[vm_id]
        if vm.active_workloads > 0:
            logger.warning(f"Cannot delete VM {vm_id}: has {vm.active_workloads} active workloads.")
            return False
        del self.vms[vm_id]
        return True

    def update_vm(self, vm_id: str, cpu_cap: float, mem_cap: float, io_cap: float, net_cap: float) -> bool:
        """Updates VM capacities safely ensuring new capacities are not below current allocations."""
        if vm_id not in self.vms:
            return False
        vm = self.vms[vm_id]
        if (cpu_cap < vm.cpu_alloc or mem_cap < vm.mem_alloc or 
            io_cap < vm.io_alloc or net_cap < vm.net_alloc):
            logger.warning(f"Unsafe capacity reduction for VM {vm_id} below current allocations.")
            return False
        vm.cpu_cap = max(0.0, cpu_cap)
        vm.mem_cap = max(0.0, mem_cap)
        vm.io_cap = max(0.0, io_cap)
        vm.net_cap = max(0.0, net_cap)
        return True

    def clone(self) -> 'SimulationEnvironment':
        """Creates an isolated simulation environment with identical VM states."""
        new_env = SimulationEnvironment()
        for vm_id, vm in self.vms.items():
            new_env.vms[vm_id] = vm.clone()
        new_env.allocations = dict(self.allocations)
        new_env.workloads = {w_id: Workload(w.workload_id, w.workload_type, w.cpu_req, w.mem_req, w.io_req, w.net_req, w.arrival_time, w.duration) for w_id, w in self.workloads.items()}
        return new_env

    def reset_allocations(self):
        """Resets all live allocations back to 0 without deleting the VM definitions."""
        for vm in self.vms.values():
            vm.cpu_alloc = 0.0
            vm.mem_alloc = 0.0
            vm.io_alloc = 0.0
            vm.net_alloc = 0.0
            vm.active_workloads = 0
        self.allocations.clear()
        self.workloads.clear()

    def get_system_state(self) -> List[dict]:
        """Returns state representations for all VMs."""
        return [vm.to_dict() for vm in self.vms.values()]
