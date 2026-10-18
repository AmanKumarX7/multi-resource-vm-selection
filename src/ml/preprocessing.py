import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List

class Preprocessor:
    """Handles feature scaling and exclusion of forbidden columns."""
    
    def __init__(self, exclude_cols: List[str] = None):
        self.exclude_cols = exclude_cols or []
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame):
        """Fits the scaler on the training data."""
        X_filtered = self._filter_columns(X)
        self.scaler.fit(X_filtered)
        self.feature_names = X_filtered.columns.tolist()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted scaler."""
        X_filtered = self._filter_columns(X)
        
        # Ensure we have the same columns as fit
        if self.feature_names is not None:
            missing = set(self.feature_names) - set(X_filtered.columns)
            if missing:
                raise ValueError(f"Missing columns in transform: {missing}")
            X_filtered = X_filtered[self.feature_names]
            
        scaled_data = self.scaler.transform(X_filtered)
        return pd.DataFrame(scaled_data, columns=X_filtered.columns, index=X_filtered.index)
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the data."""
        return self.fit(X).transform(X)
        
    def _filter_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes excluded and forbidden leakage columns."""
        X_filtered = X.copy()
        
        # Explicit excluded columns
        cols_to_drop = [c for c in self.exclude_cols if c in X_filtered.columns]
        if cols_to_drop:
            X_filtered.drop(columns=cols_to_drop, inplace=True)
            
        # Implicit leakage prevention (safeguard)
        leakage_keywords = ['resulting', 'post', 'future', 'allocated_', 'target_suitability']
        leakage_cols = [c for c in X_filtered.columns if any(kw in c.lower() for kw in leakage_keywords)]
        if leakage_cols:
            X_filtered.drop(columns=leakage_cols, inplace=True)
            
        return X_filtered
