@echo off
title AegisAI - Localhost Server (FastAPI + CUDA)
color 0b

echo ========================================================
echo    AEGIS AI - CONTEXTUAL TOXICITY MODERATION ENGINE
echo    Hardware: RTX 3050 CUDA 12.8 & Model V4
echo ========================================================
echo.
echo Starting local server on http://127.0.0.1:8000 ...
echo.
echo ========================================================
echo Web UI:      http://127.0.0.1:8000
echo API Docs:    http://127.0.0.1:8000/docs
echo ========================================================
echo (Keep this window open while using the application)
echo.

python -m uvicorn app.app:app --host 127.0.0.1 --port 8000 --reload

pause
