@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo CNN (YOLOv8) Sise Sayim Arayuzu Baslatiliyor...
echo ============================================================
streamlit run app.py
pause
