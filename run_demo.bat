@echo off
echo ========================================================
echo Intelligent Multi-Resource VM Selection: Demo Launcher
echo ========================================================
echo.

echo [1] Checking Preflight Requirements...
venv\Scripts\python scripts\preflight.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Preflight failed. Please resolve the issues above.
    pause
    exit /b %ERRORLEVEL%
)
echo.


echo [2] Running Quick Experiment (Smoke Test)...
venv\Scripts\python scripts\run_experiment.py --vms 50 --workloads 100 --seed 42
echo.

echo [3] Launching Streamlit Research Dashboard...
echo Press CTRL+C to stop the dashboard when finished.
venv\Scripts\streamlit run src\ui\dashboard.py
pause
