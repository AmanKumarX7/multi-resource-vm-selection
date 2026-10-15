import os
import sys
import logging
from datetime import datetime

# Adjust path for execution from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.dataset_generator import DatasetGenerator
from src.ml.data_loader import DataLoader
from src.ml.preprocessing import Preprocessor
from src.ml.splitting import DataSplitter
from src.ml.trainer import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.ml.model_persistence import ModelPersistence
from src.database.repositories import ExperimentRepository, ModelRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Starting ML Training Pipeline...")
    
    # 1. Generate Dataset
    logger.info("Generating dataset...")
    gen = DatasetGenerator(seed=42)
    # Using a small config for reproducible run
    csv_path = gen.generate_dataset(num_vms=20, num_workloads=1000, out_dir="data")
    
    # 2. Load Data
    logger.info("Loading dataset...")
    df = DataLoader.load_dataset(csv_path)
    
    # 3. Split Data (Preventing Leakage)
    logger.info("Splitting dataset (grouping by scenario_id)...")
    train_df, test_df, y_train, y_test = DataSplitter.split(
        df, target_col='target_suitability', scenario_col='scenario_id', test_size=0.2, random_state=42
    )
    
    # 4. Preprocess Data
    logger.info("Preprocessing data (excluding leakage columns)...")
    preprocessor = Preprocessor(exclude_cols=['scenario_id'])
    X_train_scaled = preprocessor.fit_transform(train_df)
    X_test_scaled = preprocessor.transform(test_df)
    
    # 5. Train Model
    logger.info("Training RandomForestRegressor...")
    hyperparameters = {'n_estimators': 50, 'max_depth': 10, 'random_state': 42}
    trainer = ModelTrainer(**hyperparameters)
    model = trainer.train(X_train_scaled, y_train)
    
    # 6. Evaluate Model
    logger.info("Evaluating model...")
    preds = model.predict(X_test_scaled)
    metrics = ModelEvaluator.evaluate(y_test, preds)
    for k, v in metrics.items():
        logger.info(f"Metric - {k}: {v:.4f}")
        
    # 7. Persist Model
    logger.info("Saving model pipeline...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"rf_model_{timestamp}"
    file_path = ModelPersistence.save_pipeline(model, preprocessor, "models", model_name)
    logger.info(f"Model saved to {file_path}")
    
    # 8. Log to Database
    logger.info("Logging to MySQL database...")
    try:
        exp_id = ExperimentRepository.create_experiment(
            name=f"ML_Train_{timestamp}",
            description="Milestone 6: Random Forest Training"
        )
        run_id = ModelRepository.create_model_run(
            experiment_id=exp_id,
            model_name=model_name,
            hyperparameters=str(hyperparameters)
        )
        ModelRepository.log_metrics(run_id, metrics)
        logger.info(f"Database logging complete. Run ID: {run_id}")
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")
        logger.warning("Make sure MySQL is running and configured in .env")
        
    logger.info("ML Training Pipeline finished successfully.")
    
if __name__ == "__main__":
    run_pipeline()
