@echo off
rem Downloads the BirdNET example recording (if needed) and identifies the birds in it.
rem Needs ffmpeg (see 1_setup.bat). The example recording is from New York, so we use New York coordinates.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
where ffmpeg >nul 2>nul
if errorlevel 1 goto noffmpeg
if not exist examples mkdir examples
if exist examples\soundscape.wav goto haveaudio
echo Downloading the example recording ...
curl.exe -L -f -o examples\soundscape.wav https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/main/birdnet_analyzer/example/soundscape.wav
if errorlevel 1 goto dlfail
:haveaudio
%PY% dawn_chorus.py listen examples\soundscape.wav --lat 42.45 --lon -76.50
if errorlevel 1 goto end
start "" examples\soundscape.notes.html
goto end
:noffmpeg
echo ffmpeg is not installed or this window was opened before installing it.
echo Run:  winget install Gyan.FFmpeg   then close VS Code and this window, and open them again.
goto end
:dlfail
echo Download failed. Check your internet connection and run this file again.
:end
echo.
pause
