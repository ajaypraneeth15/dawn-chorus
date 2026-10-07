@echo off
rem Dawn Chorus - Windows setup. Double-click this file, or run it from a terminal.
rem Installs the Python packages and checks that the AI runtime can actually load.
setlocal
cd /d "%~dp0.."
echo ===== Dawn Chorus: Windows setup =====
echo.
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
%PY% --version
if errorlevel 1 goto nopython

echo.
echo [1/3] Installing numpy and ai-edge-litert. This downloads about 100 MB.
%PY% -m pip install numpy ai-edge-litert
if errorlevel 1 goto pipfail

echo.
echo [2/3] Installing birdnetlib. About 66 MB. It carries the BirdNET model files.
%PY% -m pip install --no-deps birdnetlib
if errorlevel 1 goto pipfail

echo.
echo [3/3] Checking that the AI runtime loads ...
%PY% -c "from ai_edge_litert.interpreter import Interpreter; print('AI runtime: OK')"
if errorlevel 1 goto litertfail

echo.
where ffmpeg >nul 2>nul
if errorlevel 1 goto noffmpeg
echo ffmpeg: found
goto done

:noffmpeg
echo ffmpeg: NOT found. It is needed to read audio files (not for the bingo card).
echo   To install it, run this in a terminal (about 100 MB or more):
echo       winget install Gyan.FFmpeg
echo   Then CLOSE this window and VS Code completely, and open them again.
goto done

:nopython
echo.
echo Python was not found. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" on the first screen.
goto end

:pipfail
echo.
echo A pip install failed. Read the red text above. Check your internet connection and try again.
goto end

:litertfail
echo.
echo The package installed but Python could not load it. The real reason is in the red text above.
echo Common fix on Windows: install the Microsoft Visual C++ runtime, then run this file again:
echo       winget install Microsoft.VCRedist.2015+.x64
echo If it still fails, install Python 3.13 (https://www.python.org/downloads/) and run:
echo       py -3.13 -m pip install numpy ai-edge-litert
echo       py -3.13 -m pip install --no-deps birdnetlib
echo and use "py -3.13" in place of "python" from then on.
goto end

:done
echo.
echo Setup finished. Next: double-click 2_check.bat
:end
echo.
pause
