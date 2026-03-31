@echo off
echo Starting NEAT-DNFS dashboard...
echo -------------------------------

REM Change directory to the folder containing neat-dnfs-visualizer.py
cd /d "%~dp0"

REM Launch Streamlit using Python module syntax
python -m streamlit run neat-dnfs-visualizer.py

echo.
echo Streamlit has stopped.
pause
