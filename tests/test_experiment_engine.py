import pytest
import os
import pandas as pd
from src.evaluation.experiment_engine import ExperimentEngine
from src.ml.trainer import ModelTrainer
from src.ml.preprocessing import Preprocessor
from src.ml.model_persistence import ModelPersistence

class TestExperimentEngine:
    
    @pytest.fixture
    def mock_model_path(self, tmpdir):
        # Create a dummy model and preprocessor for MLSelector
        df = pd.DataFrame({
            'vm_cpu_util': [10, 20], 'vm_mem_util': [10, 20], 'vm_io_util': [10, 20], 'vm_net_util': [10, 20],
            'vm_cpu_avail': [90, 80], 'vm_mem_avail': [90, 80], 'vm_io_avail': [90, 80], 'vm_net_avail': [90, 80],
            'vm_cpu_cap': [100, 100], 'vm_mem_cap': [100, 100], 'vm_io_cap': [100, 100], 'vm_net_cap': [100, 100],
            'wl_cpu_req': [10, 10], 'wl_mem_req': [10, 10], 'wl_io_req': [10, 10], 'wl_net_req': [10, 10],
            'wl_type_enc': [0, 0]
        })
        y = [80.0, 40.0]
        
        preprocessor = Preprocessor()
        X_scaled = preprocessor.fit_transform(df)
        
        trainer = ModelTrainer(n_estimators=5, random_state=42)
        model = trainer.train(X_scaled, y)
        
        model_dir = str(tmpdir.mkdir("models"))
        return ModelPersistence.save_pipeline(model, preprocessor, model_dir, "test_rf")

    def test_reproducibility(self, mock_model_path):
        engine = ExperimentEngine(mock_model_path)
        
        # Because we only have mock workload/vms in the DB for unit testing, 
        # saving results might trigger foreign key errors. 
        # But wait, experiment_engine catches exceptions on db save and continues.
        
        base_res_1, ml_res_1 = engine.run_comparison(num_vms=5, num_workloads=10, seed=42)
        base_res_2, ml_res_2 = engine.run_comparison(num_vms=5, num_workloads=10, seed=42)
        
        assert base_res_1['allocation_successes'] == base_res_2['allocation_successes']
        assert base_res_1['allocation_failures'] == base_res_2['allocation_failures']
        assert ml_res_1['allocation_successes'] == ml_res_2['allocation_successes']
        
    def test_metrics_calculation(self, mock_model_path):
        engine = ExperimentEngine(mock_model_path)
        base_res, ml_res = engine.run_comparison(num_vms=2, num_workloads=5, seed=123)
        
        # Check that metrics exist
        expected_metrics = [
            'allocation_successes', 'allocation_failures', 'overloaded_vms',
            'cpu_balance_std', 'mem_balance_std', 'io_balance_std', 'net_balance_std', 'composite_balance_std',
            'mean_decision_time', 'median_decision_time',
            'min_decision_time', 'max_decision_time', 'std_decision_time',
            'total_sim_time', 'mean_relative_quality'
        ]
        for m in expected_metrics:
            assert m in base_res
            assert m in ml_res
            
        assert base_res['total_sim_time'] > 0
        assert ml_res['total_sim_time'] > 0
        assert base_res['mean_decision_time'] >= 0
        assert ml_res['mean_decision_time'] >= 0
