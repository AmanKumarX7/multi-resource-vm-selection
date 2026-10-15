import os
import joblib
from typing import Tuple, Any

class ModelPersistence:
    """Handles saving and loading of trained models and preprocessors."""
    
    @staticmethod
    def save_pipeline(model: Any, preprocessor: Any, 
                      model_dir: str, model_name: str) -> str:
        """Saves both model and preprocessor."""
        os.makedirs(model_dir, exist_ok=True)
        
        pipeline = {
            'model': model,
            'preprocessor': preprocessor
        }
        
        file_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(pipeline, file_path)
        return file_path
        
    @staticmethod
    def load_pipeline(file_path: str) -> Tuple[Any, Any]:
        """Loads both model and preprocessor from a joblib file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
            
        pipeline = joblib.load(file_path)
        return pipeline['model'], pipeline['preprocessor']
