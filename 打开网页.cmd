@echo off
rem ASCII only in this file: cmd.exe codepage mangles non-ASCII batch text.
rem All Chinese hints are printed by sim\server.py instead.
rem Double-click: starts the web demo in AGENT mode with the best human-data POLICY.
rem   - the LLM splits your sentence into tool calls
rem   - the ACT policy does the motion
rem   - if it fails, the real result goes back to the LLM, which picks another plan
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

set "CKPT=runs\train\act_human_104_c40\checkpoints\020000\pretrained_model"
if not exist "%CKPT%\config.json" set "CKPT=runs\train\act_grasp_v2\checkpoints\040000\pretrained_model"
if not exist "%CKPT%\config.json" goto no_policy

"%PY%" "%~dp0sim\server.py" --port 8000 --open --agent --policy "%CKPT%"
goto done

:no_policy
echo [WARN] No trained policy found. Starting IK-only demo instead.
"%PY%" "%~dp0sim\server.py" --port 8000 --open

:done
echo.
echo Done. Press any key to close this window.
pause >nul
