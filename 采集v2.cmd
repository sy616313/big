@echo off
rem ASCII only in this file: cmd.exe chokes on a UTF-8 BOM and mangles non-ASCII
rem batch text depending on the active code page.
rem Records into data\human_v2 and never touches data\human_v1.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

rem "%*" forwards any arguments, e.g. run:  collect_v2 --force
"%PY%" "%~dp0scripts\teleop.py" --out data\human_v2 %*

echo.
echo Done. Press any key to close this window.
pause >nul
