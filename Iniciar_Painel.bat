@echo off
setlocal
cd /d "%~dp0"

set "PORT=8501"
set "LOGFILE=%~dp0streamlit.log"
set "STREAMLIT_DIR=%USERPROFILE%\.streamlit"
set "CRED_FILE=%STREAMLIT_DIR%\credentials.toml"

where python >nul 2>nul
if errorlevel 1 (
  echo Python nao encontrado no PATH.
  echo Instale o Python e tente novamente.
  pause
  exit /b 1
)

if not exist "%STREAMLIT_DIR%" mkdir "%STREAMLIT_DIR%"
if not exist "%CRED_FILE%" (
  (
    echo [general]
    echo email = ""
  ) > "%CRED_FILE%"
)

echo Encerrando processo anterior na porta %PORT% (se existir)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
  taskkill /PID %%a /F >nul 2>nul
)

echo Iniciando Streamlit...
start "Painel Leads" cmd /c "cd /d ""%~dp0"" && python -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false > ""%LOGFILE%"" 2>&1"

echo Aguardando servidor subir na porta %PORT%...
set /a WAIT=0
:waitloop
set /a WAIT+=1
netstat -ano | findstr :%PORT% | findstr LISTENING >nul 2>nul
if not errorlevel 1 goto ready
if %WAIT% GEQ 30 goto fail
timeout /t 1 >nul
goto waitloop

:ready
echo Servidor ativo. Abrindo navegador...
start "" http://localhost:%PORT%
exit /b 0

:fail
echo.
echo Nao foi possivel iniciar o servidor em ate 30 segundos.
echo Veja o log em: %LOGFILE%
echo.
if exist "%LOGFILE%" type "%LOGFILE%"
pause
exit /b 1
