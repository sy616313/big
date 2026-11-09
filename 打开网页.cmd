@echo off
rem ASCII only in this file: cmd.exe codepage mangles non-ASCII batch text.
rem All Chinese hints are printed by sim\server.py instead.
rem Double-click: web demo in AGENT mode with the ENSEMBLE human-data policy.
rem   - two ACT models look at the same frame, actions are averaged
rem   - same code path as the offline evaluation (sim\ensemble.py)
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

set "CKPT_A=runs\train\act_h154_aug40k\checkpoints\040000\pretrained_model"
set "CKPT_B=runs\train\act_h104_aug\checkpoints\020000\pretrained_model"
if not exist "%CKPT_A%\config.json" goto solo
if not exist "%CKPT_B%\config.json" goto solo
"%PY%" "%~dp0sim\server.py" --port 8000 --open --agent --policy "%CKPT_A%" "%CKPT_B%"
goto done

:solo
if not exist "%CKPT_A%\config.json" set "CKPT_A=runs\train\act_grasp_v2\checkpoints\040000\pretrained_model"
if not exist "%CKPT_A%\config.json" goto no_policy
"%PY%" "%~dp0sim\server.py" --port 8000 --open --agent --policy "%CKPT_A%"
goto done

:no_policy
echo [WARN] No trained policy found. Starting IK-only demo instead.
"%PY%" "%~dp0sim\server.py" --port 8000 --open

:done
echo.
echo Done. Press any key to close this window.
pause >nul
