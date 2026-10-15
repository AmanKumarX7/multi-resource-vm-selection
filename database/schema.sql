CREATE DATABASE IF NOT EXISTS `vm_selection_research`;
USE `vm_selection_research`;

-- 1. Experiments
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. VMs
CREATE TABLE IF NOT EXISTS vms (
    internal_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    vm_id VARCHAR(100) NOT NULL,
    cpu_cap FLOAT NOT NULL,
    mem_cap FLOAT NOT NULL,
    io_cap FLOAT NOT NULL,
    net_cap FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- 3. Workloads
CREATE TABLE IF NOT EXISTS workloads (
    internal_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    workload_id VARCHAR(100) NOT NULL,
    workload_type VARCHAR(100) NOT NULL,
    cpu_req FLOAT NOT NULL,
    mem_req FLOAT NOT NULL,
    io_req FLOAT NOT NULL,
    net_req FLOAT NOT NULL,
    arrival_time INT,
    duration INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- 4. Resource Metrics
CREATE TABLE IF NOT EXISTS resource_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    vm_internal_id INT NOT NULL,
    tick INT NOT NULL,
    cpu_util FLOAT NOT NULL,
    mem_util FLOAT NOT NULL,
    io_util FLOAT NOT NULL,
    net_util FLOAT NOT NULL,
    active_workloads INT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vm_internal_id) REFERENCES vms(internal_id) ON DELETE CASCADE
);
CREATE INDEX idx_resource_metrics_vm_tick ON resource_metrics(vm_internal_id, tick);

-- 5. Allocation Decisions
CREATE TABLE IF NOT EXISTS allocation_decisions (
    decision_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    workload_internal_id INT NOT NULL,
    vm_internal_id INT, -- NULL if infeasible/rejected universally
    method VARCHAR(50) NOT NULL, -- e.g. 'baseline', 'random_forest'
    decision_time_seconds DOUBLE NOT NULL,
    is_successful BOOLEAN NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (workload_internal_id) REFERENCES workloads(internal_id) ON DELETE CASCADE,
    FOREIGN KEY (vm_internal_id) REFERENCES vms(internal_id) ON DELETE CASCADE
);
CREATE INDEX idx_alloc_decision_experiment ON allocation_decisions(experiment_id, method);

-- 6. Experiment Results
CREATE TABLE IF NOT EXISTS experiment_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- 7. Model Runs
CREATE TABLE IF NOT EXISTS model_runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT,
    model_name VARCHAR(100) NOT NULL,
    hyperparameters TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE SET NULL
);

-- 8. Model Metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE NOT NULL,
    step INT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE
);
