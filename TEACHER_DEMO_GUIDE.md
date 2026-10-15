# Teacher Demonstration Guide

This guide is designed for an external evaluator (such as a professor) to cleanly deploy and review the core components of the Intelligent Multi-Resource VM Selection project on a Windows machine.

## Prerequisites
- A Windows computer.
- Python 3.11 installed.
- MySQL Server installed and running locally on port `3306`.

## 1. Environment Setup (One-time only)
1. **Clone/Download** the repository to your local machine.
2. **Configure Database Credentials**: 
   - Open the root folder.
   - Rename `.env.example` to `.env`.
   - Edit `.env` and set your MySQL `DB_USER` and `DB_PASSWORD` (usually `root` and your password).
3. **Install Dependencies**:
   - Open Command Prompt or PowerShell in the root directory.
   - Run the following commands to create the environment and install packages:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     ```
4. **Initialize the Database**:
   - Run: `venv\Scripts\python scripts\setup_db.py`
   - *Expected Result*: "Database schema successfully enforced."
5. **Train the Initial ML Model**:
   - Run: `venv\Scripts\python scripts\train_model.py`
   - *Expected Result*: It generates a dataset, trains a Random Forest, and outputs "Pipeline saved to models/...".

## 2. Launching the Demo
You can launch the entire interactive demo sequence by simply double-clicking the **`run_demo.bat`** file from the root folder, or running it in the terminal:
```cmd
run_demo.bat
```

### What `run_demo.bat` does sequentially:
1. **Preflight Check**: Verifies Python, environment, dependencies, `.env` file, and MySQL schema availability.
2. **Quick Experiment**: Headlessly runs the `ExperimentEngine` using the Baseline and Random Forest algorithms across a 50 VM / 100 Workload simulation to populate fresh data into MySQL.
3. **Dashboard Launch**: Automatically spins up the Streamlit UI on `http://localhost:8501`. 

### Using the Dashboard:
Once the browser opens to the Dashboard:
- Review the **Methodology** outlining the specific standard-deviation balance tracking metrics.
- Use the **Run New Experiment** sidebar to trigger new custom simulations (e.g., 250 VMs, 250 Workloads).
- Scroll down to view the **Raw Metrics** database table comparing Baseline vs. ML across Allocation Quality, Balancing Standards, and fractional Decision Time.
- Observe the **Scalability Trends** line charts (populated by earlier matrix runs) illustrating the algorithmic trade-off: Random Forest scales with flat time penalties, while Baseline is faster but degrades differently in multi-resource balance at massive scale.

## 3. Running the Test Suite (Optional)
To verify all logical constraints (e.g. deterministic isolation, dataset leakage protection, SQL exception handling):
```cmd
venv\Scripts\pytest
```
*Expected Result*: `30 passed` (100% success rate without warnings).
