import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from typing import Tuple, List

class DataSplitter:
    """Handles train/test splitting, respecting scenario boundaries to prevent leakage."""
    
    @staticmethod
    def split(df: pd.DataFrame, target_col: str = 'target_suitability', 
              scenario_col: str = 'scenario_id', test_size: float = 0.2, 
              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into train and test sets.
        If scenario_col exists, it uses GroupShuffleSplit to keep candidates from the same 
        workload (scenario) together. Otherwise, does random split.
        """
        if scenario_col in df.columns:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(df, groups=df[scenario_col]))
            
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            
        y_train = train_df.pop(target_col)
        y_test = test_df.pop(target_col)
        
        # Remove scenario_col if present, as it is not a feature
        if scenario_col in train_df.columns:
            train_df.drop(columns=[scenario_col], inplace=True)
        if scenario_col in test_df.columns:
            test_df.drop(columns=[scenario_col], inplace=True)
            
        return train_df, test_df, y_train, y_test
