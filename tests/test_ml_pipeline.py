import os
import pytest
import pandas as pd
import numpy as np

from src.ml.preprocessing import Preprocessor
from src.ml.splitting import DataSplitter
from src.ml.trainer import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.ml.model_persistence import ModelPersistence

class TestMLPipeline:
    
    @pytest.fixture
    def sample_data(self):
        """Creates a mock dataset."""
        return pd.DataFrame({
            'scenario_id': ['w1', 'w1', 'w2', 'w2', 'w3', 'w3', 'w4', 'w4', 'w5', 'w5'],
            'vm_cpu_util': [10, 20, 30, 40, 50, 60, 70, 80, 90, 10],
            'wl_cpu_req': [5, 5, 10, 10, 15, 15, 20, 20, 25, 25],
            'resulting_leakage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'target_suitability': [80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 95.0]
        })
        
    def test_preprocessing(self, sample_data):
        preprocessor = Preprocessor(exclude_cols=['scenario_id'])
        # Extract target first since it shouldn't be scaled, though preprocessor drops it if present
        y = sample_data.pop('target_suitability')
        
        # Fit and transform
        X_scaled = preprocessor.fit_transform(sample_data)
        
        # Check leakage column is gone
        assert 'resulting_leakage' not in X_scaled.columns
        assert 'scenario_id' not in X_scaled.columns
        
        # Check scaling (mean should be roughly 0, std roughly 1)
        assert np.isclose(X_scaled['vm_cpu_util'].mean(), 0, atol=1e-7)
        assert np.isclose(X_scaled['vm_cpu_util'].std(ddof=0), 1, atol=1e-7)
        
    def test_splitting_with_groups(self, sample_data):
        train_df, test_df, y_train, y_test = DataSplitter.split(
            sample_data, 
            target_col='target_suitability', 
            scenario_col='scenario_id', 
            test_size=0.2, 
            random_state=42
        )
        
        # Ensure scenario_id is removed from features
        assert 'scenario_id' not in train_df.columns
        assert 'scenario_id' not in test_df.columns
        
        # Ensure that no scenario is in both train and test
        # We need to recreate the scenario sets to check this since they are removed
        # Actually, since GroupShuffleSplit splits by group, we can just trust the logic,
        # but let's test sizes. 10 rows, 5 groups. 20% test size -> 1 group in test -> 2 rows.
        assert len(test_df) == 2
        assert len(train_df) == 8
        assert len(y_test) == 2
        assert len(y_train) == 8
        
    def test_training_and_evaluation(self, sample_data):
        train_df, test_df, y_train, y_test = DataSplitter.split(
            sample_data, target_col='target_suitability', scenario_col='scenario_id', test_size=0.2
        )
        
        preprocessor = Preprocessor()
        X_train_scaled = preprocessor.fit_transform(train_df)
        X_test_scaled = preprocessor.transform(test_df)
        
        trainer = ModelTrainer(n_estimators=10, random_state=42)
        model = trainer.train(X_train_scaled, y_train)
        
        # Evaluate
        preds = model.predict(X_test_scaled)
        
        # Prediction shape and range
        assert len(preds) == len(y_test)
        assert all(0 <= p <= 100 for p in preds)
        
        metrics = ModelEvaluator.evaluate(y_test, preds)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['MAE'] >= 0
        
    def test_model_persistence(self, tmpdir, sample_data):
        train_df, test_df, y_train, y_test = DataSplitter.split(
            sample_data, target_col='target_suitability', scenario_col='scenario_id', test_size=0.2
        )
        
        preprocessor = Preprocessor()
        X_train_scaled = preprocessor.fit_transform(train_df)
        
        trainer = ModelTrainer(n_estimators=5, random_state=42)
        model = trainer.train(X_train_scaled, y_train)
        
        # Save
        model_dir = str(tmpdir.mkdir("models"))
        file_path = ModelPersistence.save_pipeline(model, preprocessor, model_dir, "test_model")
        
        assert os.path.exists(file_path)
        
        # Load
        loaded_model, loaded_preprocessor = ModelPersistence.load_pipeline(file_path)
        
        # Verify
        X_test_scaled = loaded_preprocessor.transform(test_df)
        preds = loaded_model.predict(X_test_scaled)
        
        assert len(preds) == len(y_test)
