@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [ERRO] Ambiente virtual nao encontrado. Rode instalar.bat primeiro.
    pause
    exit /b 1
)
echo Iniciando treinamento com deteccao, segmentacao e classificacao...
.venv\Scripts\python.exe treinamento.py
pause
