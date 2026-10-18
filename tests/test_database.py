import pytest
import os
from src.database.connection import DatabaseConfig, get_db_connection

def test_database_config_loading(monkeypatch):
    monkeypatch.setenv("DB_HOST", "test_host")
    monkeypatch.setenv("DB_PORT", "1234")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_pass")
    
    
    config = DatabaseConfig.load_config()
    assert config["host"] == "test_host"
    assert config["port"] == 1234
    assert config["database"] == "test_db"
    assert config["user"] == "test_user"
    assert config["password"] == "test_pass"

def test_database_config_missing_user(monkeypatch):
    monkeypatch.setenv("DB_USER", "")
    with pytest.raises(ValueError, match="Database credentials"):
        with get_db_connection() as conn:
            pass

def db_is_available():
    try:
        # Avoid checking if DB_USER isn't set, as it raises ValueError
        config = DatabaseConfig.load_config()
        if not config["user"]:
            return False
            
        with get_db_connection() as conn:
            return True
    except Exception:
        return False

@pytest.mark.skipif(not db_is_available(), reason="MySQL database is not available or configured with credentials. Skipping integration test.")
def test_experiment_repository_integration():
    from src.database.repositories import ExperimentRepository
    
    exp_id = ExperimentRepository.create_experiment("Test Experiment", "Integration test")
    assert isinstance(exp_id, int)
    assert exp_id > 0
