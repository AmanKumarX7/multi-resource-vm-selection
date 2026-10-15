import os
import pytest
import pandas as pd
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.ml.dataset_generator import DatasetGenerator

class TestDatasetGenerator:
    
    def test_ground_truth_target_validity(self):
        gen = DatasetGenerator()
        
        # Test infeasible
        vm = VM("V1", 100, 10, 1000, 1000)
        wl_inf = Workload("w1", "cpu", 101, 1, 1, 1, 0, 10)
        assert gen.calculate_ground_truth(vm, wl_inf) == 0.0
        
        # Test perfect fit (everything uses exactly 50%)
        vm2 = VM("V2", 100, 100, 100, 100)
        wl_perf = Workload("w2", "balanced", 50, 50, 50, 50, 0, 10)
        
        score = gen.calculate_ground_truth(vm2, wl_perf)
        # util = [50, 50, 50, 50]
        # max = 50. stdev = 0
        # 100 - 50 - 0 = 50
        assert score == 50.0
        
        # Ensure bounding
        vm3 = VM("V3", 100, 100, 100, 100)
        wl_max = Workload("w3", "cpu", 100, 100, 100, 100, 0, 10)
        score_max = gen.calculate_ground_truth(vm3, wl_max)
        assert score_max == 0.0

    def test_feature_extraction_leakage(self):
        gen = DatasetGenerator()
        vm = VM("V1", 100, 100, 100, 100)
        wl = Workload("w1", "cpu", 10, 20, 30, 40, 0, 10)
        
        features = gen.extract_features(vm, wl)
        
        # The word 'target' should never be in the raw extracted features dictionary
        for k in features.keys():
            assert 'target' not in k.lower()
            
        assert features['wl_cpu_req'] == 10
        assert features['vm_cpu_avail'] == 100

    def test_generation_reproducibility(self, tmpdir):
        # tmpdir is provided by pytest
        data_dir = str(tmpdir.mkdir("data"))
        
        gen1 = DatasetGenerator(seed=123)
        path1 = gen1.generate_dataset(num_vms=5, num_workloads=10, out_dir=data_dir)
        df1 = pd.read_csv(path1)
        
        gen2 = DatasetGenerator(seed=123)
        path2 = gen2.generate_dataset(num_vms=5, num_workloads=10, out_dir=data_dir)
        df2 = pd.read_csv(path2)
        
        gen3 = DatasetGenerator(seed=999)
        path3 = gen3.generate_dataset(num_vms=5, num_workloads=10, out_dir=data_dir)
        df3 = pd.read_csv(path3)
        
        # Test 1 & 2 perfectly identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Test 1 & 3 diff
        assert not df1.equals(df3)

    def test_dataset_validation_fails_on_leakage(self):
        gen = DatasetGenerator()
        
        # Mock dataframe with bad feature
        df = pd.DataFrame({
            'vm_cpu_util': [10.0],
            'resulting_cpu_util': [20.0], # LEAKAGE!
            'target_suitability': [50.0]
        })
        
        with pytest.raises(ValueError, match="leakage"):
            gen.validate_dataset(df)
            
    def test_dataset_validation_fails_on_missing(self):
        gen = DatasetGenerator()
        df = pd.DataFrame({
            'vm_cpu_util': [10.0, None],
            'target_suitability': [50.0, 50.0]
        })
        
        with pytest.raises(ValueError, match="missing values"):
            gen.validate_dataset(df)
