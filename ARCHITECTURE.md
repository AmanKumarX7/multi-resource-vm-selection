# Architecture Overview

This project implements a multi-resource VM selection simulation supporting both a rule-based baseline and a Machine Learning (Random Forest) intelligent selector. 

## Flow

1. **Workload Generator**: Generates simulated workloads requiring CPU, Memory, Disk I/O, and Network.
2. **VM / Host Simulation**: Maintains the individual simulated states of VMs across the simulation duration.

3. **Resource State Generator / Monitor**: Monitors current capacities and identifies overloads or capacities.
4. **Multi-Resource Analyzer**: Forms candidates for VM allocation.
5. **Decision Engine**: Two branching logic paths:
    - **Baseline Strategy**: Utilizes rule-based configuration weighting resources.
    - **Random Forest Strategy**: Features inputs processed via trained ML models for selection suitability.
6. **VM Selection -> Workload Allocation**: Decides the allocation step and assigns the VM.
7. **Resource Update**: Dynamically changes capacities post-allocation.
8. **MySQL Logging**: Stores VM states, transactions, metrics, and outcomes transparently.
9. **Streamlit Academic Interface**: A multi-tab dashboard/sandbox offering:
    - **Live Simulator**: Real-time exploratory console. Users can provision live VMs, queue custom workloads, and step through allocations manually comparing ML vs. Baseline. Built with Apple-inspired minimalist aesthetics.
    - **Experiments**: A headless runner that triggers the `ExperimentEngine`.
    - **Research Analysis**: Dynamic visualizer mapping MySQL aggregate results (time scale, quality trends) into Plotly charts.
    - **System Information**: Fetches underlying hardware telemetry (`psutil`) to prove sandbox isolation.
