@echo off
cd /d "%~dp0.."
python -m streamlit run Dashboardpp.py --browser.gatherUsageStats false
