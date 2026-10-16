"""
Standardized Physical and Simulation Units.

Defines the project-wide measurement units for VM capacities and Workload demands:
- CPU: MHz (e.g., 1000 MHz capacity, 100 MHz request)
- Memory: GB (e.g., 16 GB capacity, 2 GB request)
- Disk I/O: MB/s (e.g., 5000 MB/s capacity, 500 MB/s request)
- Network Throughput: MB/s (e.g., 1000 MB/s capacity, 100 MB/s request)
- Workload Duration: Simulation ticks (e.g., 10 ticks)
- Decision Time: Milliseconds (ms)
"""

class ResourceUnits:
    CPU = "MHz"
    RAM = "GB"
    DISK_IO = "MB/s"
    NET = "MB/s"
    DURATION = "ticks"
    DECISION_TIME = "ms"

    # Input form and UI display labels
    CPU_CAP_LABEL = f"CPU Capacity ({CPU})"
    RAM_CAP_LABEL = f"RAM Capacity ({RAM})"
    DISK_IO_CAP_LABEL = f"Disk I/O Capacity ({DISK_IO})"
    NET_CAP_LABEL = f"Network Capacity ({NET})"

    CPU_REQ_LABEL = f"CPU Request ({CPU})"
    RAM_REQ_LABEL = f"RAM Request ({RAM})"
    DISK_IO_REQ_LABEL = f"Disk I/O Request ({DISK_IO})"
    NET_REQ_LABEL = f"Network Request ({NET})"
    DURATION_LABEL = f"Duration ({DURATION})"

    @staticmethod
    def format_cpu(value: float) -> str:
        return f"{value:,.0f} {ResourceUnits.CPU}"

    @staticmethod
    def format_ram(value: float) -> str:
        return f"{value:,.1f} {ResourceUnits.RAM}"

    @staticmethod
    def format_io(value: float) -> str:
        return f"{value:,.0f} {ResourceUnits.DISK_IO}"

    @staticmethod
    def format_net(value: float) -> str:
        return f"{value:,.0f} {ResourceUnits.NET}"

    @staticmethod
    def format_duration(ticks: int) -> str:
        return f"{ticks} {ResourceUnits.DURATION}"

    @staticmethod
    def format_decision_time_ms(seconds: float) -> str:
        ms = seconds * 1000.0
        return f"{ms:.3f} {ResourceUnits.DECISION_TIME}"

    @staticmethod
    def format_decision_time(seconds: float) -> str:
        ms = seconds * 1000.0
        return f"{ms:.4f} {ResourceUnits.DECISION_TIME}"

    @staticmethod
    def get_unit_badge(resource: str) -> str:
        badges = {
            "cpu": ResourceUnits.CPU,
            "mem": ResourceUnits.RAM,
            "ram": ResourceUnits.RAM,
            "io": ResourceUnits.DISK_IO,
            "net": ResourceUnits.NET,
            "duration": ResourceUnits.DURATION,
            "decision_time": ResourceUnits.DECISION_TIME
        }
        return badges.get(resource.lower(), "")
