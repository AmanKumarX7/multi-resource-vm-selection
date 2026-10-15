import pandas as pd
import json
import os
from typing import Tuple, Dict, Any

class DataLoader:
    """Loads datasets and metadata for ML training."""
    
    @staticmethod
    def load_dataset(csv_path: str) -> pd.DataFrame:
        """Loads a generated dataset CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        return pd.read_csv(csv_path)
        
    @staticmethod
    def load_metadata(json_path: str) -> Dict[str, Any]:
        """Loads dataset metadata JSON."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata not found at {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
