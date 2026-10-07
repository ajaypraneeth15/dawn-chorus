@echo off
rem Runs the automated tests. No Ollama, no internet and no model downloads needed.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
%PY% -m unittest discover -s tests -v
echo.
pause
