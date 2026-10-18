# Investigating the Trade-off Between Load Balancing Accuracy and Decision Time in Multi-Resource VM Selection

<<<<<<< HEAD
A simulation-based research project that investigates the trade-off between load-balancing quality and decision time in multi-resource Virtual Machine (VM) selection. The project compares a baseline resource-aware selection method with a Machine Learning-based Random Forest approach across CPU, memory, disk I/O, and network resources.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [How the Project Works](#how-the-project-works)
- [Methods Compared](#methods-compared)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running Experiments](#running-experiments)
- [Launching the Live Dashboard](#launching-the-live-dashboard)
- [Running Tests](#running-tests)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing Status](#testing-status)
- [Limitations](#limitations)
- [Team Members](#team-members)
- [Project Documentation](#project-documentation)
- [Future Improvements](#future-improvements)

## Introduction

Virtualization enables multiple virtual machines to share the resources of a physical computing system. However, inefficient workload allocation can cause some VMs to become overloaded while others remain underutilized.

Load balancing helps distribute workloads across available VMs. In a multi-resource environment, selecting an appropriate VM requires considering several factors, including CPU utilization, memory consumption, disk I/O, and network utilization.

This project investigates how different VM-selection strategies perform when balancing resource utilization while keeping decision-making time efficient.

The project integrates concepts from:

- Operating Systems
- Database Management Systems (DBMS)
- Machine Learning
- Virtualization and Load Balancing
- Algorithm Design and Performance Evaluation

The system provides a simulation environment for comparing VM-selection approaches under configurable workloads.

## Problem Statement

Traditional VM selection based on limited resource information may not always produce balanced workload distribution. A VM with sufficient CPU capacity may still be unsuitable if its memory, disk, or network resources are heavily utilized.

At the same time, more sophisticated selection strategies may require additional computation, increasing decision time as the number of VMs and workloads grows.

The primary research question is:

**How can multi-resource VM selection balance resource utilization while maintaining efficient decision-making as the simulated environment scales?**

This project studies the relationship between load-balancing quality and decision time using simulation-based experiments.

## Project Objectives

1. Investigate how intelligent VM selection can improve resource utilization balance.
2. Compare a baseline resource-aware selection method with a Random Forest-based approach.
3. Analyze decision time under different numbers of VMs and workloads.
4. Evaluate multiple resource dimensions instead of relying exclusively on CPU utilization.
5. Support repeatable experiments through configurable parameters and random seeds.
6. Integrate database persistence for storing and inspecting experiment data.
7. Provide an interactive dashboard for demonstrating and exploring the simulation.

## How the Project Works

The project follows a simulation and evaluation workflow.

1. **Simulation configuration:** Define the number of virtual machines, workloads, and random seed.
2. **Resource representation:** Model resource-related information for CPU, memory, disk I/O, and network utilization.
3. **VM selection:** Execute the implemented baseline and Random Forest-based selection approaches.
4. **Performance measurement:** Collect the metrics supported by the implementation, including decision time and load-balancing-related measurements.
5. **Data persistence:** Use the database integration to store experiment information where configured.
6. **Result analysis:** Compare the approaches across different simulation configurations.
7. **Interactive visualization:** Use the Streamlit dashboard to explore and demonstrate the simulator.

The objective is to understand how selection quality and computational cost change as the simulated workload increases.

## Methods Compared

### 1. Baseline Resource-Aware Selection

The baseline approach uses resource-related information to make VM-selection decisions. It provides a reference against which the Machine Learning-based approach can be evaluated.

### 2. Random Forest-Based Selection

Random Forest is a supervised Machine Learning algorithm that combines predictions from multiple decision trees.

In this project, the trained Random Forest model is used as part of the VM-selection comparison. The training workflow generates the required dataset and model artifacts used by subsequent experiments.

### Research Focus

The project focuses on two important considerations:

- **Load-balancing quality:** How effectively the method distributes resource demand.
- **Decision time:** How long the method takes to make a VM-selection decision.

A method that improves load distribution may also introduce additional computational overhead. Therefore, both dimensions should be considered when interpreting experimental results.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Simulation, experiment execution, and ML workflow |
| Random Forest | Machine Learning-based VM selection |
| MySQL | Database and experiment data persistence |
| Streamlit | Interactive simulator and dashboard |
| pytest | Automated testing |
| Git | Version control |
| GitHub | Source code hosting and team collaboration |

The complete list of Python dependencies is available in `requirements.txt`.

## Repository Structure

```text
multi-resource-vm-selection/
│
├── database/                 # Database-related files
│
├── scripts/
│   ├── setup_db.py            # Database initialization
│   ├── train_model.py         # Dataset generation and model training
│   └── run_experiment.py      # Command-line experiment runner
│
├── src/
│   └── ui/
│       └── dashboard.py       # Streamlit dashboard
│
├── tests/                     # Automated test suite
│
├── .env.example               # Example environment configuration
├── .gitignore                 # Git exclusion rules
├── ARCHITECTURE.md             # Architecture documentation
├── PROJECT_STATUS.md           # Project implementation status
├── README.md                   # Main project documentation
├── TEACHER_DEMO_GUIDE.md       # Project demonstration guide
├── requirements.txt            # Python dependencies
└── run_demo.bat                # Windows demo launcher
```

This structure describes the main project components. Refer to the repository for the latest file organization.

## Prerequisites

Before installing the project, ensure that you have:

- Python 3.11 installed.
- MySQL Server installed and running if using database-backed features.
- Git installed for cloning the repository.
- A Windows terminal such as Command Prompt or PowerShell.

## Installation and Setup

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/sharma-ayushi22/multi-resource-vm-selection.git
```

Navigate to the project directory:

```bash
cd multi-resource-vm-selection
```

### Step 2: Create a Virtual Environment

Create the Python virtual environment:

```bash
python -m venv venv
```

Activate it using Command Prompt:

```bat
venv\Scripts\activate
```

Alternatively, in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Database

Create a `.env` file in the project root directory.

Add the following configuration and replace the placeholder password with your own MySQL password:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=yourpassword
```

**Security note:** Never commit your real database password or other secrets to GitHub. Keep `.env` private and use `.env.example` for placeholder configuration.

### Step 5: Initialize the Database

Ensure that MySQL Server is running, then execute:

```bash
python scripts/setup_db.py
```

This runs the project's database initialization script.

If a connection error occurs, verify your MySQL service, credentials, port, and database permissions.

### Step 6: Train the Machine Learning Model

Before running experiments that depend on the trained model, execute:

```bash
python scripts/train_model.py
```

This runs the project's model training workflow and generates the required artifacts according to the implementation.

## Running Experiments

Run a sample command-line experiment:

```bash
python scripts/run_experiment.py --vms 50 --workloads 100 --seed 42
```

### Experiment Parameters

| Parameter | Description |
|---|---|
| `--vms 50` | Number of virtual machines in the simulation |
| `--workloads 100` | Number of workloads to simulate |
| `--seed 42` | Random seed for repeatability where supported |

You can modify the VM and workload counts to investigate how the selection methods behave at different scales.

For example, you can compare a smaller simulation with a larger one:

```bash
python scripts/run_experiment.py --vms 10 --workloads 20 --seed 42
```

```bash
python scripts/run_experiment.py --vms 100 --workloads 200 --seed 42
```

These are example configurations, not reported experimental results.

To view all supported command-line arguments, run:

```bash
python scripts/run_experiment.py --help
```

## Launching the Live Dashboard

The project includes a Streamlit-based live simulator and research dashboard.

Start the dashboard from the project root:

```bash
python -m streamlit run src/ui/dashboard.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the interface.

Make sure the required dependencies, database configuration, and trained model artifacts are available if the dashboard depends on them.

## Running Tests

The project includes automated tests for different parts of the implementation.

Run the test suite using:

```bash
python -m pytest -v
```

The test suite covers areas including:

- Simulation behavior and bounds
- Selection algorithms
- Machine Learning pipeline
- Database persistence
- Dashboard and UI integration

Review the test output and traceback if a test fails.

## Evaluation Metrics

The project's research objective is to investigate the trade-off between load-balancing quality and decision time.

The following dimensions are relevant to the analysis:

### 1. Load-Balancing Quality

Measures how effectively workloads are distributed across VMs according to the metrics implemented in the simulator.

### 2. Decision Time

Measures the time taken by a selection method to choose a VM.

### 3. Scalability

Examines how the methods behave as the number of VMs and workloads increases.

### 4. Repeatability

Uses consistent experiment configurations and random seeds where supported to help make comparisons reproducible.

### 5. Multi-Resource Consideration

Examines resource-related information across CPU, memory, disk I/O, and network rather than considering only one resource.

**Important:** Use the actual metric definitions, units, and outputs produced by the source code when reporting findings. Numerical performance claims should be supported by measured experimental results.

## Testing Status

The project currently reports **48 passing tests** across unit tests, algorithms, the Machine Learning pipeline, database persistence, and UI integration.

This is the previously reported test status. Run the test suite on the current version of the repository to verify it.

## Limitations

- The project is a simulation-based study and does not automatically establish performance in a real cloud environment.
- Results depend on the workload model, resource assumptions, dataset, and implementation.
- Execution time can vary across machines and software environments.
- A fixed random seed supports repeatability where randomness is controlled, but does not guarantee identical execution times.
- Any conclusion about which approach performs better should be supported by actual experimental measurements.

## Team Members

This project was developed collaboratively by the following team:

| Team Member | Role | Student ID |
|---|---|---|
| Parash Rastogi | Team Lead | 240211699 |
| Aman Kumar | Team Member | 24011614 |
| Ayushi Sharma | Team Member | 24012538 |

## Project Documentation

Additional project documentation is available in the repository:

- **[ARCHITECTURE.md](ARCHITECTURE.md):** Architecture and design details.
- **[PROJECT_STATUS.md](PROJECT_STATUS.md):** Current implementation status and milestones.
- **[TEACHER_DEMO_GUIDE.md](TEACHER_DEMO_GUIDE.md):** Instructions for demonstrating the project.
- **[requirements.txt](requirements.txt):** Python dependencies.

## Future Improvements

Possible future extensions include:

- Conducting experiments across a wider range of VM and workload counts.
- Repeating experiments to analyze variability in results.
- Adding charts for load-balancing metrics and decision time.
- Documenting dataset generation and model evaluation in greater detail.
- Comparing additional VM scheduling and selection algorithms.
- Testing with real workload traces or a cloud environment where resources are available.

These are potential improvements rather than claims about features already implemented.

## Contributing

Contributions and improvements are welcome.

1. Create a new branch for your changes.
2. Implement a focused improvement or fix.
3. Run the relevant tests.
4. Commit your changes with a descriptive message.
5. Open a pull request explaining the changes.

## Academic Note

This project is intended for academic learning, simulation, and research into multi-resource VM selection. Its findings should be evaluated using reproducible experiments, clearly defined metrics, and documented assumptions.

---

**Developed collaboratively by the project team.**
=======
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
>>>>>>> 8af0194b3d3165cf0cb04d2aac74b0b34f667eb8
