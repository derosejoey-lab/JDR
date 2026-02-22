@echo off
echo ============================================================
echo   FUNDAMENTAL SCORING TOOL
echo   Scoring your stock universe... this will take 15-30 min
echo   for ~595 stocks (it pulls live data for each one).
echo
echo   You'll see progress updates as it goes.
echo   When done, your results will be saved as a CSV file.
echo ============================================================
echo.

python fundamental_scorer.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Make sure you ran setup_windows.bat first.
    echo If the issue persists, check your internet connection.
)

echo.
echo Press any key to close this window...
pause >nul
