import sys
import os
import importlib.util

def check_python():
    print("[*] Checking Python version...")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
        print("  [!] WARNING: Python 3.11 is recommended. Current version is", sys.version.split(" ")[0])
    else:
        print("  [OK] Python 3.11 detected.")

def check_venv():
    print("\n[*] Checking Virtual Environment...")
    if sys.prefix == sys.base_prefix:
        print("  [!] WARNING: Not running inside a virtual environment.")
        return False
    else:
        print("  [OK] Running inside a virtual environment.")
        return True

def check_packages():
    print("\n[*] Checking required packages...")
    packages = ["mysql.connector", "dotenv", "tabulate", "streamlit", "plotly", "pandas", "sklearn"]
    all_ok = True
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            print(f"  [X] Missing package: {pkg}")
            all_ok = False
        else:
            print(f"  [OK] Package found: {pkg}")
    return all_ok

def check_env():
    print("\n[*] Checking .env file...")
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        print("  [OK] .env file found.")
        return True
    else:
        print("  [X] .env file is missing.")
        return False

def check_mysql():
    print("\n[*] Checking MySQL Connectivity and Schema...")
    from dotenv import load_dotenv
    import mysql.connector
    load_dotenv()
    
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        print("  [OK] Connected to MySQL Server.")
        
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES LIKE 'vm_selection_research'")
        if not cursor.fetchone():
            print("  [X] Database 'vm_selection_research' does not exist. Run scripts/setup_db.py.")
            return False
            
        conn.database = 'vm_selection_research'
        cursor.execute("SHOW TABLES")
        tables = [t[0] for t in cursor.fetchall()]
        required_tables = ["experiments", "vms", "workloads", "allocation_decisions", "model_runs", "model_metrics", "experiment_results"]
        missing = [t for t in required_tables if t not in tables]
        
        if missing:
            print(f"  [X] Missing tables: {missing}. Run scripts/setup_db.py.")
            return False
            
        print("  [OK] Database and all required tables exist.")
        return True
    except Exception as e:
        print(f"  [X] MySQL connection failed: {e}")
        return False

def check_models():
    print("\n[*] Checking Trained Models...")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    if not os.path.exists(models_dir):
        print("  [X] 'models' directory not found.")
        return False
        
    joblibs = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not joblibs:
        print("  [X] No trained model found. Run scripts/train_model.py.")
        return False
        
    print(f"  [OK] Found {len(joblibs)} trained model(s).")
    return True

def main():
    print("=== System Preflight Verification ===\n")
    check_python()
    check_venv()
    
    if not check_packages():
        print("\n[!] Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
        
    if not check_env():
        print("\n[!] Please create a .env file with your MySQL credentials.")
        sys.exit(1)
        
    db_ok = check_mysql()
    model_ok = check_models()
    
    print("\n=== Preflight Results ===")
    if db_ok and model_ok:
        print("All checks PASSED. The system is ready to launch experiments or dashboard.")
    else:
        print("Some checks FAILED. Please resolve the issues above.")
        
if __name__ == "__main__":
    main()
