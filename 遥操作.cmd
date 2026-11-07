@echo off
rem ASCII only in this file: cmd.exe chokes on UTF-8 BOM and mangles non-ASCII
rem batch text depending on the active code page. All Chinese hints are printed
rem by scripts\teleop.py instead.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0scripts\teleop.py"

echo.
echo Done. Press any key to close this window.
pause >nul
