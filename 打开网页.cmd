@echo off
rem ASCII only in this file: cmd.exe codepage mangles non-ASCII batch text.
rem All Chinese hints are printed by sim\server.py instead.
rem Double-click: web demo in AGENT mode with the demo-corrected ACT policy.
rem   data\human_fix (grasp points aligned to the cube centre) -> 78% on the
rem   old four seed sets, 65% on the fresh ones, and fast enough for live play.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

set "CKPT=runs\train\human_fix_40k\checkpoints\040000\pretrained_model"
if not exist "%CKPT%\config.json" set "CKPT=runs\train\act_h154_aug40k\checkpoints\040000\pretrained_model"
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
