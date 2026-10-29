@echo off
title AegisAI - Neural Toxicity Predictor (Live Server + Public Tunnel)
color 0b

echo ========================================================
echo    AEGIS AI - CONTEXTUAL TOXICITY MODERATION ENGINE
echo    Hardware: RTX 3050 CUDA 12.8 & Model V4
echo ========================================================
echo.

echo [1/2] Starting local FastAPI server on http://127.0.0.1:8000...
start /b python -m uvicorn app.app:app --host 127.0.0.1 --port 8000

timeout /t 3 /nobreak > nul

echo [2/2] Launching Cloudflare Public Secure HTTPS Tunnel...
echo.
echo ========================================================
echo Look for the 'https://*.trycloudflare.com' URL below!
echo Share that URL with anyone to access your live model!
echo ========================================================
echo.

.\cloudflared.exe tunnel --url http://127.0.0.1:8000

pause
