@echo off
chcp 65001 >nul
cd /d "%~dp0"
python ngrok_canli_baslat.py
pause
