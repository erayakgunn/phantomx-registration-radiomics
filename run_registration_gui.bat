@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "Kod\registration-radiomics\gui.py"
if errorlevel 1 pause
