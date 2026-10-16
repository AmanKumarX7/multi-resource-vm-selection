# Project Status

**Current Milestone**: Milestone 12

## Completed Tasks
- [x] Inspected the local work environment (Python 3.11.9, Git)
- [x] Created project structure
- [x] Implemented simulated VM model with bounded capacities (`src/simulation/vm.py`)
- [x] Implemented Workload model mapping to resource profiles (`src/simulation/workload.py`)
- [x] Implemented reproducible Workload Generator using normal distributions (`src/simulation/workload_generator.py`)
- [x] Implemented Simulation Environment supporting allocation loops (`src/simulation/environment.py`)
- [x] Designed and passed unit tests via pytest (`tests/test_simulation.py`)
- [x] Implemented deterministic Baseline VM Selector (`src/algorithms/baseline.py`)
- [x] Added `time.perf_counter()` to measure strict algorithm decision time
- [x] Wrote tests to strictly assess decision-time returns and configurable deterministic matching (`tests/test_algorithms.py`)
- [x] Drafted unified relational MySQL schema representing resources, decisions, workloads, VMs, and machine-learning execution bindings (`database/schema.sql`)
- [x] Created configurable DB contexts (`src/database/connection.py`) parsing strictly via `.env` injection
- [x] Abstracted secure DB CRUD mechanics into `repositories.py` preventing injections and monolith loops
- [x] Wrote robust environment setup script `scripts/setup_db.py`
- [x] Engineered ML dataset generator isolating pre-allocation features to prevent target leakage (`src/ml/dataset_generator.py`)
- [x] Derived and implemented an independent Ground-Truth Suitability metric prioritizing multi-resource balance
- [x] Exported dataset with dynamic runtime validation checks and saved generation tracking parameters via metadata JSON
- [x] Defined and trained RandomForestRegressor model as the target `target_suitability` is a continuous metric.
- [x] Developed separate modules for ML tasks: data loading, preprocessing, split handling (by scenario_id), training, evaluation, and persistence.
- [x] Used `GroupShuffleSplit` for data splitting to avoid leakage among candidate VMs for the same scenario.
- [x] Established automated ML pipeline script (`scripts/train_model.py`) recording reproducible configuration and metrics to MySQL tables (`model_runs` & `model_metrics`).
- [x] Added rigorous pytest suite for the ML pipeline components ensuring no-leak preprocessing, prediction ranges, and pipeline save/load mechanics.
- [x] Built a reproducible fair-comparison `ExperimentEngine` ensuring both selectors run against identically seeded VM/workload environments.
- [x] Logged comprehensive metrics (resource balance, strict decision times, allocation success, relative selection quality) separate from framework I/O overhead.
- [x] Executed via CLI runner `scripts/run_experiment.py` recording true results to `experiments` and `experiment_results` MySQL tables.
- [x] **Corrective Patch**: Fixed foreign-key lockouts by robustly inserting workloads/VMs into the DB prior to allocation evaluation, ensuring no exceptions are suppressed.
- [x] **Corrective Patch**: Expanded the `resource_balance_std` metric into five specific metrics: `cpu_balance_std`, `mem_balance_std`, `io_balance_std`, `net_balance_std`, and a unified `composite_balance_std`.
- [x] Built a professional Streamlit dashboard (`src/ui/dashboard.py`) connecting directly to MySQL repositories.
- [x] Implemented UI controls for defining reproducible experiments (VM/Workload counts, seeds) running the actual backend `ExperimentEngine`.
- [x] Created responsive `Plotly` visualizations for Decision Time vs Selection Quality and unified composite multi-resource balancing.
- [x] Built automated UI integration tests via `AppTest` that simulate button clicks and execute complete DB-persisted pipeline evaluations without spinning up a browser.
- [x] **Milestone 10**: Conducted comprehensive codebase audit. Resolved MySQL connector warnings (`stored_results`), eliminated silent exceptions in repository queries, verified `.gitignore` strictness, parameterized all CRUD bindings, and formalized `README.md` guide. 
- [x] Passed 30/30 unit test assertions cleanly without warnings.
- [x] Built `ScalabilityRunner` aggregating VM-count and Workload-count scalability test matrices securely.
- [x] Created `run_scalability.py` supporting `quick` and `standard` CLI profiles cleanly generating repeated aggregated results.
- [x] Updated Streamlit UI mapping `Plotly` graphs natively visualizing these aggregate matrix arrays.
- [x] **Milestone 12**: Created strict preflight validation script (`scripts/preflight.py`) to verify environment capabilities cleanly.
- [x] Created seamless `run_demo.bat` to package preflight, smoke testing, and UI dashboard spinning into a double-click experience.
- [x] Drafted `TEACHER_DEMO_GUIDE.md` and finalized repository-friendly instruction paths.
- [x] **Phase-II Milestone — Live Simulator Redesign & Functional Integration**:
  - [x] Redesigned `src/ui/dashboard.py` into a dark infrastructure-management console with clean block layout.
  - [x] Placed Algorithm Execution Mode (`Compare Both`, `Baseline Only`, `Random Forest Only`) at the top of the Live Simulator.
  - [x] Enforced state isolation via independent cloned environments (`env_baseline`, `env_rf`) ensuring 100% fair comparison from equivalent initial states.
  - [x] Standardized physical resource units across the whole project (`src/simulation/units.py`): CPU (MHz), RAM (GB), Disk I/O (MB/s), Network (MB/s), Duration (ticks), Decision Time (ms).
  - [x] Full MySQL CRUD persistence for VMs (`vms`), workloads (`workloads`), allocation decisions (`allocation_decisions`), and simulation events (`simulation_events`) surviving application restart.
  - [x] Built live side-by-side comparison panels with honest decision-time reporting in ms and explicit error handling for missing/infeasible models.
  - [x] Added duration-based simulation tick advancement with automatic resource expiration and release.
  - [x] Built live comparative visualizations (Per-VM multi-resource utilization comparison, decision latency, cumulative outcomes).
  - [x] Built MySQL Persistence Transparency panel with live table record inspection for teacher evaluation.
  - [x] Fixed root cause of Reproducible Experiments error (`MLSelector object has no attribute pipeline_path`).
  - [x] Added comprehensive unit & integration tests (`tests/test_live_simulator.py`) bringing passing test count to 48/48 (100% pass rate).

## Known Issues
- None.

## Next Milestone
Milestone 13: Finalize and Generate the Formal Academic Research Report.

## Manual Steps Required By User
- Launch dashboard: `.\venv\Scripts\python.exe -m streamlit run src/ui/dashboard.py`

