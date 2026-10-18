import os
import logging
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
class DatabaseConfig:
    @staticmethod
    def load_config() -> dict:
        load_dotenv()
        return
        {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 3306)),
            "database": os.getenv("DB_NAME", "vm_selection_research"),
            "user": os.getenv("DB_USER", ""),
            "password": os.getenv("DB_PASSWORD", ""),
        }

@contextmanager
def get_db_connection():
    """
    Context manager that yields a database connection.
    Automatically closes upon finishing block.
    """
    config = DatabaseConfig.load_config()
    if not config["user"]:
        raise ValueError("Database credentials (DB_USER) are missing in environment.")
        connection = None

    try:
        connection = mysql.connector.connect
        (
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"]
        )
    
        if connection.is_connected():
            yield connection
        else:
            raise ConnectionError("Failed to connect to MySQL database.")
    except Exception as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise


    finally:
        if connection and connection.is_connected():
            connection.close()
