@echo off
echo ============================================================
echo   FUNDAMENTAL SCORING TOOL - SETUP
echo   This will install everything you need. Takes about 2 min.
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed on your computer.
    echo.
    echo Please install Python first:
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Click the big yellow "Download Python" button
    echo   3. IMPORTANT: Check the box that says "Add Python to PATH"
    echo   4. Click "Install Now"
    echo   5. After it finishes, close this window and double-click setup_windows.bat again
    echo.
    pause
    exit /b 1
)

echo Python found! Installing required packages...
echo.

pip install yfinance pandas numpy --upgrade
if errorlevel 1 (
    echo.
    echo There was an issue installing packages. Trying with python -m pip...
    python -m pip install yfinance pandas numpy --upgrade
)

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo   Now double-click "run_scoring.bat" to generate your ratings.
echo ============================================================
echo.
pause
