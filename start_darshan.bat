@echo off
title Darshan QML Suite

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Run Darshan
start cmd /k "python darshan.py"
