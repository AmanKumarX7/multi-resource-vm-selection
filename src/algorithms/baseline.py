import time
import logging
from typing import List, Tuple, Optional

from src.simulation.vm import VM
from src.simulation.workload import Workload

logger = logging.getLogger(__name__)

class BaselineSelector:
    """
    A deterministic, rule-based VM selection baseline approach without ML.
    It ranks feasible VMs based on a normalized weighted sum of their resultant utilizations
    if the incoming workload were to be allocated. Higher resultant utilization == higher score.
    The algorithm selects the feasible VM with the LOWEST resulting score (preferring least loaded).
    """

    def __init__(self, w_cpu: float = 0.25, w_mem: float = 0.25, w_io: float = 0.25, w_net: float = 0.25):
        # Configure normalized weights safely
        total_w = w_cpu + w_mem + w_io + w_net
        if total_w <= 0:
            logger.warning("Total weight is zero or negative. Falling back to uniform weights.")
            total_w = 4.0
            w_cpu, w_mem, w_io, w_net = 1.0, 1.0, 1.0, 1.0
            
        self.weights = {
            'cpu': w_cpu / total_w,
            'mem': w_mem / total_w,
            'io': w_io / total_w,
            'net': w_net / total_w
        }

    def _calculate_score(self, vm: VM, wl: Workload) -> float:
        """
        Calculates the theoretical hypothetical score if the workload was placed on the VM.
        Uses resultant utilization.
        Score = w_cpu * CPU_util + w_mem * Mem_util + w_io * IO_util + w_net * Net_util
        """
        # calculate resulting allocation
        cpu_alloc = vm.cpu_alloc + wl.cpu_req
        mem_alloc = vm.mem_alloc + wl.mem_req
        io_alloc = vm.io_alloc + wl.io_req
        net_alloc = vm.net_alloc + wl.net_req

        # calculate resulting utilization percentage
        cpu_u = (cpu_alloc / vm.cpu_cap * 100.0) if vm.cpu_cap > 0 else 100.0
        mem_u = (mem_alloc / vm.mem_cap * 100.0) if vm.mem_cap > 0 else 100.0
        io_u = (io_alloc / vm.io_cap * 100.0) if vm.io_cap > 0 else 100.0
        net_u = (net_alloc / vm.net_cap * 100.0) if vm.net_cap > 0 else 100.0

        score = (
            self.weights['cpu'] * cpu_u +
            self.weights['mem'] * mem_u +
            self.weights['io'] * io_u +
            self.weights['net'] * net_u
        )
        return score

    def select_vm(self, feasible_vms: List[VM], workload: Workload) -> Tuple[Optional[VM], float]:
        """
        Selects the best VM for the given workload out of a pre-filtered list of feasible VMs.
        The algorithm itself doesn't check feasibility but expects the caller to do so.
        Returns a tuple of (Selected_VM, decision_time_in_seconds).
        """
        start_time = time.perf_counter()

        if not feasible_vms:
            decision_time = time.perf_counter() - start_time
            return None, decision_time

        scored_vms = []
        for vm in feasible_vms:
            score = self._calculate_score(vm, workload)
            scored_vms.append((score, vm.vm_id, vm))

        # Tie-breaker rule:
        # Sort based on calculated score (ascending). 
        # If score is exactly identical, sort by vm.vm_id alphabetically.
        scored_vms.sort(key=lambda x: (x[0], x[1]))
        
        selected_vm = scored_vms[0][2]
        
        decision_time = time.perf_counter() - start_time
        return selected_vm, decision_time
