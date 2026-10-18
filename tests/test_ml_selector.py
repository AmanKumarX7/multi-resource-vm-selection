import pytest
import os
import pandas as pd
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.algorithms.ml_selector import MLSelector

from src.ml.trainer import ModelTrainer
from src.ml.preprocessing import Preprocessor
from src.ml.model_persistence import ModelPersistence

class TestMLSelector:
    
    @pytest.fixture
    def mock_model_path(self, tmpdir):
        # Create a dummy model and preprocessor
        df = pd.DataFrame({
            'vm_cpu_util': [10, 20], 'vm_mem_util': [10, 20], 'vm_io_util': [10, 20], 'vm_net_util': [10, 20],
            'vm_cpu_avail': [90, 80], 'vm_mem_avail': [90, 80], 'vm_io_avail': [90, 80], 'vm_net_avail': [90, 80],
            'vm_cpu_cap': [100, 100], 'vm_mem_cap': [100, 100], 'vm_io_cap': [100, 100], 'vm_net_cap': [100, 100],
            'wl_cpu_req': [10, 10], 'wl_mem_req': [10, 10], 'wl_io_req': [10, 10], 'wl_net_req': [10, 10],
            'wl_type_enc': [0, 0]
        })
        y = [80.0, 40.0] # Dummy target values
        
        preprocessor = Preprocessor()
        X_scaled = preprocessor.fit_transform(df)
        
        trainer = ModelTrainer(n_estimators=5, random_state=42)
        model = trainer.train(X_scaled, y)
        
        model_dir = str(tmpdir.mkdir("models"))
        return ModelPersistence.save_pipeline(model, preprocessor, model_dir, "test_rf")

    def test_ml_selector_initialization_fails_gracefully(self):
        with pytest.raises(FileNotFoundError):
            MLSelector(model_path="non_existent_path.joblib")


    def test_ml_selector_no_feasible_vms(self, mock_model_path):
        selector = MLSelector(mock_model_path)
        wl = Workload("w1", "balanced", 50, 5, 100, 100, 0, 10)
        vm, dt = selector.select_vm([], wl)
        
        assert vm is None
        assert dt > 0

    def test_ml_selector_selection_highest_suitability(self, mock_model_path):
        selector = MLSelector(mock_model_path)
        
        # Two identical capacity VMs, one loaded one empty.
        vm_loaded = VM("VM_A", 100, 100, 100, 100)
        vm_loaded.allocate(90, 90, 90, 90) # Only 10 avail
        
        vm_empty = VM("VM_B", 100, 100, 100, 100) # 100 avail
        
        wl = Workload("w1", "balanced", 5, 5, 5, 5, 0, 10)
        feasible = [vm_loaded, vm_empty]
        
        selected, dt = selector.select_vm(feasible, wl)
        
        # The ML should prefer the empty one as it results in better suitability
        assert selected is not None
        assert selected.vm_id == "VM_B"
        assert dt > 0

    def test_ml_selector_deterministic_tie_breaking(self, mock_model_path):
        selector = MLSelector(mock_model_path)
        
        # Create identical VMs
        vm_c = VM("VM_C", 100, 100, 100, 100)
        vm_a = VM("VM_A", 100, 100, 100, 100)
        vm_b = VM("VM_B", 100, 100, 100, 100)
        
        wl = Workload("w1", "balanced", 5, 5, 5, 5, 0, 10)
        feasible = [vm_c, vm_b, vm_a]
        
        selected, dt = selector.select_vm(feasible, wl)
        
        # Since features are identical, predictions will be identical.
        # Tie breaker should sort by VM ID ascending.
        assert selected.vm_id == "VM_A"

    def test_ml_selector_no_allocation_side_effects(self, mock_model_path):
        selector = MLSelector(mock_model_path)
        vm_a = VM("VM_A", 100, 100, 100, 100)
        wl = Workload("w1", "balanced", 5, 5, 5, 5, 0, 10)
        
        # Before selection
        cpu_before = vm_a.cpu_alloc
        
        selected, _ = selector.select_vm([vm_a], wl)
        
        # After selection, the selector should just evaluate, not allocate.
        assert selected.vm_id == "VM_A"
        assert vm_a.cpu_alloc == cpu_before

    def test_decision_time_isolated(self, mock_model_path):
        selector = MLSelector(mock_model_path)
        vm_a = VM("VM_A", 100, 100, 100, 100)
        wl = Workload("w1", "balanced", 5, 5, 5, 5, 0, 10)
        
        selected, dt = selector.select_vm([vm_a], wl)
        
        assert isinstance(dt, float)
        assert dt > 0.0
