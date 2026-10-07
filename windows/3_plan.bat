@echo off
rem Makes the bingo card for where you are this week, then opens it in your browser.
setlocal
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
echo Find your latitude and longitude: in Google Maps, right-click your spot and click the numbers.
echo Just press Enter to use the example place (Cayuga Lake, New York).
echo.
set LAT=42.45
set LON=-76.50
set /p LAT=Latitude  [42.45]: 
set /p LON=Longitude [-76.50]: 
%PY% dawn_chorus.py plan --lat %LAT% --lon %LON% --place "My walk"
if errorlevel 1 goto end
start "" bingo.html
:end
echo.
pause
