import os
import sys
import mysql.connector
from mysql.connector.constants import ClientFlag
from dotenv import load_dotenv

def setup_database():
    load_dotenv()
    
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DB_USER", "")
    db_password = os.getenv("DB_PASSWORD", "")
    
    if not db_user:
        print("ERROR: DB_USER is not set. Please ensure you have configured your .env file according to .env.example with credentials.")
        sys.exit(1)
        
    try:
        conn = mysql.connector.connect(
            host=db_host, 
            port=db_port, 
            user=db_user, 
            password=db_password,
            client_flags=[ClientFlag.MULTI_STATEMENTS]
        )
        cursor = conn.cursor()
        
        schema_path = os.path.join(os.path.dirname(__file__), "..", "database", "schema.sql")
        if not os.path.exists(schema_path):
            print(f"ERROR: Schema file not found at {schema_path}")
            sys.exit(1)
            
        with open(schema_path, "r") as f:
            sql_script = f.read()
            
        # Execute the script cleanly via multi-statements flag on the connection
        cursor.execute(sql_script)
        
        # We must read through all generated results (even if empty) to clear the connection state
        while cursor.nextset():
            pass
        
        conn.commit()
        print("Database schema successfully enforced.")
        
    except mysql.connector.Error as e:
        print(f"MySQL Error: {e}")
        print("Please check your database credentials and ensure the server is running.")
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    setup_database()
