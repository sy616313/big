@echo off
rem ASCII only in this file: cmd.exe codepage mangles non-ASCII batch text.
rem All Chinese hints are printed by sim\server.py instead.
rem Double-click: starts the local web demo and opens the browser.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0sim\server.py" --port 8000 --open

echo.
echo Done. Press any key to close this window.
pause >nul
