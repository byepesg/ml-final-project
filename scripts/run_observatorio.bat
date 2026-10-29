@echo off
REM Ejecuta el Observatorio Electoral (Streamlit) en Windows desde la raiz del repositorio.
REM Uso:
REM   scripts\run_observatorio.bat
REM   scripts\run_observatorio.bat --server.port 8600
setlocal

REM Ir a la raiz del repositorio (carpeta superior a la del script).
cd /d "%~dp0.."

REM Activar el entorno virtual del proyecto si existe y no hay otro activo.
if not defined VIRTUAL_ENV (
    if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
)

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit no esta instalado en este entorno.
    echo Instala las dependencias con:  pip install -r requirements.txt
    exit /b 1
)

echo Iniciando el observatorio (Ctrl+C para detenerlo)...
python -m streamlit run observatorio\app.py %*
