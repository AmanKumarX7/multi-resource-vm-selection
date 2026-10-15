import logging

logger = logging.getLogger(__name__)

class VM:
    """
    Simulated Virtual Machine containing resource capacities and utilizations.
    Capacity represents the absolute resources (e.g., 1000 units CPU).
    Allocation represents how many units are currently used.
    """

    def __init__(self, vm_id: str, cpu_cap: float, mem_cap: float, io_cap: float, net_cap: float):
        self.vm_id = vm_id
        # Capacities
        self.cpu_cap = max(0.0, cpu_cap)
        self.mem_cap = max(0.0, mem_cap)
        self.io_cap = max(0.0, io_cap)
        self.net_cap = max(0.0, net_cap)
        
        # Current Allocations
        self.cpu_alloc = 0.0
        self.mem_alloc = 0.0
        self.io_alloc = 0.0
        self.net_alloc = 0.0

        self.active_workloads = 0
        self.status = "Active"

    def can_allocate(self, cpu_req: float, mem_req: float, io_req: float, net_req: float) -> bool:
        """Checks if the VM has enough remaining capacity for the given requirements."""
        if self.status != "Active":
            return False
            
        cpu_ok = (self.cpu_cap - self.cpu_alloc) >= cpu_req
        mem_ok = (self.mem_cap - self.mem_alloc) >= mem_req
        io_ok = (self.io_cap - self.io_alloc) >= io_req
        net_ok = (self.net_cap - self.net_alloc) >= net_req
        
        return cpu_ok and mem_ok and io_ok and net_ok

    def allocate(self, cpu_req: float, mem_req: float, io_req: float, net_req: float) -> bool:
        """Allocates resources if feasible."""
        if not self.can_allocate(cpu_req, mem_req, io_req, net_req):
            logger.warning(f"VM {self.vm_id} cannot allocate requested resources. Overload risk.")
            return False
            
        self.cpu_alloc += cpu_req
        self.mem_alloc += mem_req
        self.io_alloc += io_req
        self.net_alloc += net_req
        self.active_workloads += 1
        return True

    def release(self, cpu_req: float, mem_req: float, io_req: float, net_req: float):
        """Releases resources and ensures allocations do not drop below zero."""
        self.cpu_alloc = max(0.0, self.cpu_alloc - cpu_req)
        self.mem_alloc = max(0.0, self.mem_alloc - mem_req)
        self.io_alloc = max(0.0, self.io_alloc - io_req)
        self.net_alloc = max(0.0, self.net_alloc - net_req)
        self.active_workloads = max(0, self.active_workloads - 1)

    def _calc_util(self, alloc: float, cap: float) -> float:
        if cap <= 0:
            return 100.0 if alloc > 0 else 0.0
        # Bound utilization between 0% and 100%
        return max(0.0, min(100.0, (alloc / cap) * 100.0))

    @property
    def cpu_utilization(self) -> float:
        return self._calc_util(self.cpu_alloc, self.cpu_cap)

    @property
    def memory_utilization(self) -> float:
        return self._calc_util(self.mem_alloc, self.mem_cap)

    @property
    def io_utilization(self) -> float:
        return self._calc_util(self.io_alloc, self.io_cap)

    @property
    def network_utilization(self) -> float:
        return self._calc_util(self.net_alloc, self.net_cap)

    def to_dict(self) -> dict:
        """Returns the VM state as a dictionary."""
        return {
            'vm_id': self.vm_id,
            'cpu_util': self.cpu_utilization,
            'mem_util': self.memory_utilization,
            'io_util': self.io_utilization,
            'net_util': self.network_utilization,
            'active_workloads': self.active_workloads,
            'status': self.status
        }
