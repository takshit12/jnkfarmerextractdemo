@echo off
echo Starting AgriStack Web UI...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run Streamlit
streamlit run app.py --server.port 8501 --server.address localhost

pause
