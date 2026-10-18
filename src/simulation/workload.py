class Workload:
    """
    Represents a generalized computational job or process (workload).
    Requirements represent the absolute resource units needed. 
    Types can be: cpu-intensive, memory-intensive, io-intensive, network-intensive, balanced
    """

    def __init__(self, workload_id: str,
                 workload_type: str,
                 cpu_req: float,
                 mem_req: float,
                 io_req: float,
                 net_req: float,
                 arrival_time: int,
                 duration: int):
        self.workload_id = workload_id
        self.workload_type = workload_type
        
        # Resource requests mapping to absolute units
        self.cpu_req = max(0.0, cpu_req)
        self.mem_req = max(0.0, mem_req)
        self.io_req = max(0.0, io_req)
        self.net_req = max(0.0, net_req)
        
        # Scheduling / lifecycle markers
        self.arrival_time = arrival_time
        self.duration = duration

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "type": self.workload_type,
            "cpu_req": self.cpu_req,
            "mem_req": self.mem_req,
            "io_req": self.io_req,
            "net_req": self.net_req,
            "arrival_time": self.arrival_time,
            "duration": self.duration
        }
