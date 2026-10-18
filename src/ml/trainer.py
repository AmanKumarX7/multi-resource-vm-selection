from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any

class ModelTrainer:
    """Trains the RandomForest model for VM Suitability."""
    
    def __init__(self, **hyperparameters):
        # Default hyperparameters, easily overridden
        self.params = {
            'n_estimators': hyperparameters.get('n_estimators', 100),
            'max_depth': hyperparameters.get('max_depth', None),
            'random_state': hyperparameters.get('random_state', 42),
            'n_jobs': hyperparameters.get('n_jobs', -1)
        }
        self.model = RandomForestRegressor(**self.params)
        
    def train(self, X_train, y_train):
        """Fits the Random Forest Regressor on training data."""
        self.model.fit(X_train, y_train)
        return self.model
        
    def get_params(self) -> Dict[str, Any]:
        """Returns the configuration for logging."""
        return self.params
