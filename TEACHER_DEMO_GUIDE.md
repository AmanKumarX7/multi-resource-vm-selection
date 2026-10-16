# Teacher Demonstration Guide — Phase-II Live Interactive Simulator

This guide is designed for an external evaluator, professor, or project mentor to review the Phase-II implementation of the research project:
**"Investigating the Trade-off Between Load Balancing Accuracy and Decision Time in Multi-Resource VM Selection"**

---

## 1. System Requirements & Quick Start

### Prerequisites
- Windows 10/11
- Python 3.11.9
- MySQL Server 8.0+ running locally on port `3306`

### Launch the Application
Run the Streamlit console using the project's virtual environment:
```powershell
.\venv\Scripts\python.exe -m streamlit run src/ui/dashboard.py
```
Or double-click `run_demo.bat` in the project root directory.

---

## 2. Live Simulator Demonstration Sequence (Step-by-Step)

The primary focus of Phase-II is the **Live Interactive Simulator** tab, redesigned as a professional dark infrastructure-management console.

### Step 1: System Ribbon & Connection Verification
- Look at the top status ribbon:
  - **MySQL Status**: Confirms live connectivity to `vm_selection_research` (queried genuinely from MySQL).
  - **ML Model**: Displays loaded artifact name (`rf_model_*.joblib`) and verification state.
  - **Live Metrics**: Shows active simulation tick, provisioned VM count, and queue size.

### Step 2: Algorithm Mode Selection (Prominent Top Section)
- Located immediately below the header:
  - **Compare Both — Baseline vs Random Forest** (Default): Evaluates each incoming workload against two independent, isolated simulation states starting from equivalent initial conditions.
  - **Baseline Only**: Deterministic, weighted multi-resource least-loaded heuristic.
  - **Random Forest Only**: ML regressor predicting multi-resource ground truth suitability.
- *Explain to teacher*: State isolation prevents the "shared-state bias" where allocating to one algorithm changes the capacity seen by the other.

### Step 3: Infrastructure State & Standardized Physical Units
- Check the VM cards in the responsive grid. Capacities and demands are standardized:
  - **CPU**: MHz (e.g., 2,000 MHz capacity)
  - **RAM**: GB (e.g., 16.0 GB capacity)
  - **Disk I/O**: MB/s (e.g., 5,000 MB/s throughput)
  - **Network**: MB/s (e.g., 1,000 MB/s throughput)
  - **Duration**: Simulation ticks (e.g., 10 ticks)
  - **Decision Time**: Milliseconds (ms)
- Notice dynamic status badges (`Active`, `Near-Cap`, `Overloaded`) and dual-pool toggle (Baseline state vs Random Forest state).

### Step 4: Live Allocation Comparison (Side-by-Side Dual Panels)
- Under **Workload Management**, click **"Load 10 Standard Research Workloads"** to populate the inventory.
- Click **"Enqueue Selected Workload"** or create a custom workload.
- In **Live Execution Controls**, click **"Step (Allocate Next)"**:
  - **Panel 1 (Baseline Scheduler)**: Shows selected VM, feasible candidates, microsecond-measured decision time in milliseconds (`~0.01-0.05 ms`), and selection rationale.
  - **Panel 2 (Random Forest Selector)**: Shows selected VM, predicted suitability ranking, ML inference latency (`~20-35 ms`), and outcome.
- *Explain to teacher*: Demonstrates the core research trade-off—Baseline is ultra-fast with microsecond heuristics; Random Forest performs deep multidimensional balancing with an inference latency cost.

### Step 5: Comparative Visualizations & Analytics
- Expand **📊 Live Simulation Visualizations & Comparison Charts**:
  - **Per-VM Resource Utilization Chart**: Direct side-by-side grouped bar chart comparing CPU, RAM, Disk I/O, and Network utilization between Baseline and Random Forest.
  - **Latest Decision Latency Chart**: Visualizes the decision-time gap in milliseconds.
  - **Cumulative Allocation Outcomes**: Stacked bars showing successful allocations vs rejections.

### Step 6: Duration-Based Lifecycle & Resource Release
- In **Live Execution Controls**, click **"Step Tick (+1 Tick)"**:
  - Observe simulation tick counter advance (`Tick: 1 -> 2 -> ...`).
  - Workloads reaching their configured duration automatically release their reserved CPU, RAM, Disk I/O, and Network resources back to the VM pool.
  - Workload release events are logged to the session activity log and MySQL.

### Step 7: MySQL Persistence & Restart Verification (Key Requirement)
1. In **VM Management**, provision a new VM (e.g., `Node_Demo_Alpha`, CPU: `2500 MHz`, RAM: `16 GB`).
2. Scroll to the **MySQL Persistence & Database Transparency** panel at the bottom:
   - Check the genuine database counters (`VM Records`, `Workload Records`, `Decisions Logged`, `Simulation Events`).
   - Expand **"Teacher Demonstration: Inspect Genuine MySQL Table Rows"**:
   - Select `vms` or `workloads` or `simulation_events` to view the live rows directly queried from MySQL via parameterized SQL.
3. **Application Restart Test**:
   - Stop Streamlit (`Ctrl+C` in terminal) and re-launch:
     ```powershell
     .\venv\Scripts\python.exe -m streamlit run src/ui/dashboard.py
     ```
   - Refresh browser: confirm that `Node_Demo_Alpha` and all saved workloads remain in the pool without data loss!

---

## 3. Other Research Navigation Tabs

- **Experiments Tab**:
  - Executes headlessly decoupled matrix comparisons using `ExperimentEngine`.
  - The previously reported `'MLSelector' object has no attribute 'pipeline_path'` error is verified resolved at root cause.
- **Research Analysis Tab**:
  - Pulls historical experiment data from MySQL and plots Mean Decision Time Comparison, Multi-Resource Balance Standard Deviation, and Scalability Trends.
- **System Information Tab**:
  - Displays host machine hardware specs (sandbox execution environment) and live MySQL server telemetry.

---

## 4. Test Suite Verification

Run the automated test suite to demonstrate 100% test coverage and regression protection:
```powershell
.\venv\Scripts\python.exe -m pytest -v
```
**Expected Outcome**: 48 passed tests across 10 test modules (33 original baseline tests + 15 new Phase-II live simulator unit & integration tests).
