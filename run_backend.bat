@echo off
echo Starting AgriStack Backend Pipeline (CLI Mode)...
echo.

REM Activate virtual environment with error checking
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found! Run setup.py first.
    pause
    exit /b 1
)

REM Set default input/output if not provided
set INPUT_FILE="TransliteradVersion_Village Gujral - Jamabandi (1).pdf"
set OUTPUT_FILE="output/backend_processed_gujral.xlsx"

echo Processing: %INPUT_FILE%
echo Output to: %OUTPUT_FILE%
echo.

REM Run the main pipeline module
python -m src.main --input %INPUT_FILE% --output %OUTPUT_FILE% --pages "1" -v

echo.
echo Processing Complete!
pause
