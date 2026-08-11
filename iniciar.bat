@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [ERRO] Ambiente virtual nao encontrado. Rode instalar.bat primeiro.
    pause
    exit /b 1
)

if not exist config.cfg (
    echo [ERRO] config.cfg nao encontrado. Copie config.cfg.example para
    echo config.cfg e preencha os dados da camera antes de continuar.
    pause
    exit /b 1
)

echo Iniciando pagina de treinamento...
.venv\Scripts\python.exe web_treinamento.py

echo.
echo O processo foi encerrado.
pause
