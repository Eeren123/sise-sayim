@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo Canli Kamera Sise Sayim Penceresi Baslatiliyor...
echo Cikis icin kamera penceresindeyken 'Q' veya 'ESC' tusuna basin.
echo ============================================================
python canli_kamera.py
pause
