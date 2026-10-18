import os
import pytest
from src.evaluation.scalability import ScalabilityRunner
from src.ml.trainer import ModelTrainer
from src.ml.preprocessing import Preprocessor
from src.ml.model_persistence import ModelPersistence

import pandas as pd

class TestScalability:
    
    @pytest.fixture
    def mock_model_path(self, tmpdir):
        df = pd.DataFrame({
            'vm_cpu_util': [10, 20], 'vm_mem_util': [10, 20], 'vm_io_util': [10, 20], 'vm_net_util': [10, 20],
            'vm_cpu_avail': [90, 80], 'vm_mem_avail': [90, 80], 'vm_io_avail': [90, 80], 'vm_net_avail': [90, 80],
            'vm_cpu_cap': [100, 100], 'vm_mem_cap': [100, 100], 'vm_io_cap': [100, 100], 'vm_net_cap': [100, 100],
            'wl_cpu_req': [10, 10], 'wl_mem_req': [10, 10], 'wl_io_req': [10, 10], 'wl_net_req': [10, 10],
            'wl_type_enc': [0, 0]
        })
        y = [80.0, 40.0]
        preprocessor = Preprocessor()
        X = preprocessor.fit_transform(df)
        trainer = ModelTrainer(n_estimators=5, random_state=42)
        model = trainer.train(X, y)
        model_dir = str(tmpdir.mkdir("models"))
        return ModelPersistence.save_pipeline(model, preprocessor, model_dir, "test_rf")


    def test_aggregation_logic(self, mock_model_path):
        runner = ScalabilityRunner(mock_model_path)
        dummy_runs = [
            {'allocation_successes': 10, 'mean_decision_time': 0.1, 'total_sim_time': 1.0},
            {'allocation_successes': 20, 'mean_decision_time': 0.3, 'total_sim_time': 2.0},
            {'allocation_successes': 30, 'mean_decision_time': 0.2, 'total_sim_time': 3.0}
        ]
        agg = runner.aggregate_results(dummy_runs)
        
        
        assert agg['allocation_successes_mean'] == 20
        assert agg['allocation_successes_min'] == 10
        assert agg['allocation_successes_max'] == 30
        assert agg['mean_decision_time_mean'] == 0.2
        assert agg['total_sim_time_median'] == 2.0
        assert 'mean_decision_time_std' in agg

    def test_scalability_matrix_run(self, mock_model_path, tmpdir):
        runner = ScalabilityRunner(mock_model_path)
        runner.output_dir = str(tmpdir) # Override output for test
        
        # Run a very tiny matrix
        runner.run_matrix("Test_Matrix", vms_list=[2], workloads_list=[2], seeds=[42, 43])
        
        # Verify files generated
        assert os.path.exists(os.path.join(str(tmpdir), "Test_Matrix_raw.csv"))
        assert os.path.exists(os.path.join(str(tmpdir), "Test_Matrix_raw.json"))
        
        # Verify JSON content
        import json
        with open(os.path.join(str(tmpdir), "Test_Matrix_raw.json")) as f:
            data = json.load(f)
            
        assert len(data) == 4 # 2 seeds * 2 methods
        assert data[0]['vms'] == 2
        assert data[0]['workloads'] == 2
