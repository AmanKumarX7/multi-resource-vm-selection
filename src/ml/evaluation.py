import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

class ModelEvaluator:
    """Evaluates regression models for VM Suitability."""
    
    @staticmethod
    def evaluate(y_true, y_pred) -> Dict[str, float]:
        """Calculates regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
