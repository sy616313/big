@echo off
rem ASCII only in this file: cmd.exe codepage mangles non-ASCII batch text.
rem All Chinese hints are printed by sim\server.py instead.
rem Double-click: web demo in AGENT mode.
rem   primary policy : human_fix_40k (grasp points aligned to cube centre)
rem   retry policies : A and B (DIFFERENT models -- the policy is deterministic,
rem                    replaying the same one in the same scene does nothing)
chcp 65001 >nul
cd /d "%~dp0"

set "PY=D:\envs\embodied\python.exe"
if not exist "%PY%" set "PY=python"

set "CKPT=runs\train\human_fix_40k\checkpoints\040000\pretrained_model"
if not exist "%CKPT%\config.json" set "CKPT=runs\train\act_h154_aug40k\checkpoints\040000\pretrained_model"
if not exist "%CKPT%\config.json" set "CKPT=runs\train\act_grasp_v2\checkpoints\040000\pretrained_model"
if not exist "%CKPT%\config.json" goto no_policy

set "RT_A=runs\train\act_h154_aug40k\checkpoints\040000\pretrained_model"
set "RT_B=runs\train\act_h104_aug\checkpoints\020000\pretrained_model"
if not exist "%RT_A%\config.json" goto single
if not exist "%RT_B%\config.json" goto single
"%PY%" "%~dp0sim\server.py" --port 8000 --open --agent --policy "%CKPT%" --retry-with "%RT_A%" "%RT_B%"
goto done

:single
"%PY%" "%~dp0sim\server.py" --port 8000 --open --agent --policy "%CKPT%"
goto done

:no_policy
echo [WARN] No trained policy found. Starting IK-only demo instead.
"%PY%" "%~dp0sim\server.py" --port 8000 --open

:done
echo.
echo Done. Press any key to close this window.
pause >nul
