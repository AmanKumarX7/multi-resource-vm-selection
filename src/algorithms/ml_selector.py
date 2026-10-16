import time
import logging
import pandas as pd
from typing import List, Tuple, Optional

from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.ml.model_persistence import ModelPersistence
from src.ml.dataset_generator import DatasetGenerator

logger = logging.getLogger(__name__)

class MLSelector:
    """
    Machine Learning based VM selector using a pre-trained RandomForest model.
    It predicts the suitability score for each feasible VM candidate and chooses the highest.
    """
    
    def __init__(self, model_path: Optional[str] = None, pipeline_path: Optional[str] = None):
        self.model_path = model_path or pipeline_path
        if not self.model_path:
            raise ValueError("A valid model_path or pipeline_path must be provided.")
            
        try:
            self.model, self.preprocessor = ModelPersistence.load_pipeline(self.model_path)
            logger.info(f"Successfully loaded ML pipeline from {self.model_path}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load ML model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading ML model: {e}")
            raise
            
        self.feature_extractor = DatasetGenerator(seed=0) # Need just the extract_features method

    @property
    def pipeline_path(self) -> str:
        """Consistent accessor for model pipeline path."""
        return self.model_path

    def select_vm(self, feasible_vms: List[VM], workload: Workload) -> Tuple[Optional[VM], float]:
        """
        Selects the best VM for the given workload using the ML pipeline.
        Returns a tuple of (Selected_VM, decision_time_in_seconds).
        """
        start_time = time.perf_counter()
        
        if not feasible_vms:
            decision_time = time.perf_counter() - start_time
            return None, decision_time
            
        # 1. Extract features for all feasible VMs
        records = []
        for vm in feasible_vms:
            # We must use EXACTLY the same feature extraction logic
            features = self.feature_extractor.extract_features(vm, workload)
            records.append(features)
            
        df_candidates = pd.DataFrame(records)
        
        # 2. Preprocess (scaling, dropping any unwanted cols implicitly handled by preprocessor)
        try:
            X_scaled = self.preprocessor.transform(df_candidates)
        except Exception as e:
            logger.error(f"Error during preprocessing candidates: {e}")
            raise
            
        # 3. Predict suitability
        try:
            predictions = self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error during ML prediction: {e}")
            raise
            
        # 4. Rank candidates (tie-break logic)
        scored_vms = []
        for i, vm in enumerate(feasible_vms):
            score = predictions[i]
            scored_vms.append((score, vm.vm_id, vm))
            
        # Tie-breaker rule: 
        # Sort based on calculated score (DESCENDING, highest suitability wins)
        # If score is exactly identical, sort by vm.vm_id (ASCENDING, alphabetical)
        # Note: sorting by -score for descending, x[1] for ascending
        scored_vms.sort(key=lambda x: (-x[0], x[1]))
        
        selected_vm = scored_vms[0][2]
        decision_time = time.perf_counter() - start_time
        
        return selected_vm, decision_time
