@echo off
rem Optional: a short chat with a Gemma model (running in Ollama on this PC) before you go for a walk.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
where ollama >nul 2>nul
if errorlevel 1 goto noollama
echo Find your latitude and longitude: in Google Maps, right-click your spot and click the numbers.
echo Just press Enter to use the example place (Cayuga Lake, New York).
echo.
set LAT=42.45
set LON=-76.50
set /p LAT=Latitude  [42.45]: 
set /p LON=Longitude [-76.50]: 
echo.
echo Press Enter on an empty line when you are ready to go outside.
%PY% dawn_chorus.py companion --lat %LAT% --lon %LON% --place "My walk"
goto end
:noollama
echo Ollama was not found. Install it from https://ollama.com, open the Ollama app,
echo then download a Gemma model:  ollama pull gemma3:1b
:end
echo.
pause
