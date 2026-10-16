# Investigating the Trade-off Between Load Balancing Accuracy and Decision Time in Multi-Resource VM Selection

## Team
- Parash Rastogi — Team Lead — Student ID: 240211699
- Aman Kumar — Student ID: 24011614
- Ayushi Sharma — Student ID: 24012538

## Project Objectives
1. To investigate how intelligent multi-resource load-balancing decisions can maintain balanced resource utilization across VMs.
2. To study how efficiently VM selection decisions can be made as the number of VMs/workloads scales dynamically.

## Overview
This is a simulation-based research project integrating Operating Systems, DBMS, and Machine Learning concepts. We evaluate a baseline resource-aware VM selection model against a Random Forest approach to analyze the trade-off between load-balancing accuracy and decision time across multiple resources (CPU, Memory, Disk I/O, Network).

## Setup Instructions
1. **Python Environment**:
   Ensure you have Python 3.11 installed. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Database Setup**:
   Create a `.env` file at the project root with the following (update credentials as needed):
   ```
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=yourpassword
   ```
   Initialize the database:
   ```bash
   venv\Scripts\python scripts/setup_db.py
   ```
3. **Train the ML Model**:
   Before running experiments, generate the dataset and train the Random Forest model:
   ```bash
   venv\Scripts\python scripts/train_model.py
   ```

## Run Experiments
Run a command-line comparison experiment:
```bash
venv\Scripts\python scripts/run_experiment.py --vms 50 --workloads 100 --seed 42
```

## Run Live Simulator Dashboard
To launch the Phase-II Live Interactive Simulator and research console:
```bash
.\venv\Scripts\python.exe -m streamlit run src/ui/dashboard.py
```

## Testing
To run the full test suite (including simulation bounds, deterministic selectors, ML pipelines, MySQL persistence, and Live Simulator integration):
```bash
.\venv\Scripts\python.exe -m pytest -v
```
*(48 tests passing across unit, algorithm, ML pipeline, database persistence, and UI integration modules).*
