import random
from typing import List, Dict, Tuple
from .workload import Workload

class WorkloadGenerator:
    """
    Generates reproducible simulated workloads based on specified workload types.
    Configurations are derived around an assumed 'standard' VM resource pool, 
    but return absolute resource units mapping to requirements. 
    """

    # Assumed standard resources for relative sampling ranges
    # Standard CPU=1000 (e.g., 1000mCores), Mem=16GB, IO=5000(IOPS), Net=1000(Mbps)
    
    PROFILES = {
        'cpu-intensive': {
            'cpu': (400, 800),
            'mem': (1, 4),
            'io': (100, 500),
            'net': (10, 50)
        },
        'memory-intensive': {
            'cpu': (50, 200),
            'mem': (8, 14),
            'io': (100, 500),
            'net': (10, 50)
        },
        'io-intensive': {
            'cpu': (50, 200),
            'mem': (1, 4),
            'io': (2000, 4500),
            'net': (10, 100)
        },
        'network-intensive': {
            'cpu': (100, 300),
            'mem': (1, 4),
            'io': (100, 500),
            'net': (500, 950)
        },
        'balanced': {
            'cpu': (200, 400),
            'mem': (4, 8),
            'io': (500, 1500),
            'net': (100, 300)
        }
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(self.seed)
    
    def generate_workload_of_type(self, w_id: str, w_type: str, arrival_time: int, duration: int) -> Workload:
        bounds = self.PROFILES.get(w_type, self.PROFILES['balanced'])
        
        cpu_req = self._rng.uniform(*bounds['cpu'])
        mem_req = self._rng.uniform(*bounds['mem'])
        io_req = self._rng.uniform(*bounds['io'])
        net_req = self._rng.uniform(*bounds['net'])
        
        return Workload(
            workload_id=w_id,
            workload_type=w_type,
            cpu_req=cpu_req,
            mem_req=mem_req,
            io_req=io_req,
            net_req=net_req,
            arrival_time=arrival_time,
            duration=duration
        )

    def generate_workloads(self, count: int, type_distribution: Dict[str, float] = None) -> List[Workload]:
        """
        Generates a sequence of workloads matching the given count and type distribution.
        Distribution specifies the relative frequency of each workload category.
        """
        if not type_distribution:
            # Default distribution if none specified (20% each)
            type_distribution = {k: 0.2 for k in self.PROFILES.keys()}
        
        types = list(type_distribution.keys())
        weights = list(type_distribution.values())

        workloads = []
        for i in range(count):
            # Select workload type based on distribution weights
            chosen_type = self._rng.choices(types, weights=weights, k=1)[0]
            
            w_id = f"WL_{i}"
            arrival_time = i  # Simple sequential arrival
            duration = self._rng.randint(10, 100) # Arbitrary duration lifecycle

            wl = self.generate_workload_of_type(w_id, chosen_type, arrival_time, duration)
            workloads.append(wl)
            
        return workloads
