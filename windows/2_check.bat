@echo off
rem Checks that the BirdNET models load and nothing touches the network.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
%PY% dawn_chorus.py doctor
echo.
pause
