@echo off
rem Optional: same demo, plus a journal paragraph written by a Gemma model running in Ollama on this PC.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
where ollama >nul 2>nul
if errorlevel 1 goto noollama
if not exist examples\soundscape.wav goto noaudio
set MODEL=
for /f "tokens=1" %%a in ('ollama list ^| findstr /i gemma') do if not defined MODEL set MODEL=%%a
if not defined MODEL goto nogemma
echo Using Gemma model: %MODEL%
echo %MODEL% | findstr /i "cloud" >nul
if not errorlevel 1 echo WARNING: a model ending in -cloud runs on Ollama's servers, not on this PC.
echo This can take 1 to 2 minutes the first time.
%PY% dawn_chorus.py listen examples\soundscape.wav --lat 42.45 --lon -76.50 --journal --ollama-model %MODEL%
if errorlevel 1 goto end
start "" examples\soundscape.notes.html
goto end
:noollama
echo Ollama was not found. Install it from https://ollama.com and open the Ollama app.
goto end
:noaudio
echo Run 4_demo.bat first (it downloads the example recording).
goto end
:nogemma
echo No Gemma model is installed. Download one (about 1 GB, so use a good connection):
echo     ollama pull gemma3:1b
echo Then run this file again.
:end
echo.
pause
