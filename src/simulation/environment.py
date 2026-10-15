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

    def get_system_state(self) -> List[dict]:
        """Returns state representations for all VMs."""
        return [vm.to_dict() for vm in self.vms.values()]
