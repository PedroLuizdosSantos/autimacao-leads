@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
  taskkill /PID %%a /F >nul 2>nul
)
echo Processos da porta 8501 finalizados.
pause
