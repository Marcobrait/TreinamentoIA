@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  Instalacao - Treinamento de Modelo
echo ============================================

where python >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH. Instale o Python 3.12 e tente novamente.
    pause
    exit /b 1
)

if not exist .venv (
    echo Criando ambiente virtual em .venv ...
    python -m venv .venv
)

set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

echo Atualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [AVISO] Falha por certificado SSL nao confiavel ^(comum em rede
    echo corporativa com proxy/firewall que faz inspecao SSL^). Tentando de
    echo novo confiando diretamente em pypi.org e files.pythonhosted.org...
    .venv\Scripts\python.exe -m pip install --upgrade pip %TRUSTED_HOSTS%
)

echo Instalando dependencias do requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [AVISO] Falha por certificado SSL nao confiavel. Tentando de novo
    echo confiando diretamente em pypi.org e files.pythonhosted.org...
    .venv\Scripts\python.exe -m pip install -r requirements.txt %TRUSTED_HOSTS%
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar dependencias mesmo ignorando o certificado.
        pause
        exit /b 1
    )
)

echo.
echo Verificando se ha placa de video NVIDIA...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Nenhuma GPU NVIDIA detectada ^(comando nvidia-smi nao encontrado^).
    echo Mantendo o PyTorch para CPU ja instalado.
) else (
    echo GPU NVIDIA detectada! Trocando o PyTorch para a versao com suporte
    echo a CUDA ^(assim o treinamento usa a placa de video em vez do processador^)...
    .venv\Scripts\python.exe -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo [AVISO] Falha por certificado SSL nao confiavel. Tentando de novo
        echo confiando diretamente em download.pytorch.org...
        .venv\Scripts\python.exe -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org
        if errorlevel 1 (
            echo [AVISO] Nao foi possivel instalar o PyTorch com CUDA ^(veja o
            echo README.md, secao "GPU NVIDIA" para instalar manualmente com a
            echo versao certa para essa placa/driver^). O programa continua
            echo funcionando normalmente, so que usando a CPU.
        )
    )
)

echo.
echo ============================================
echo  Instalacao concluida.
echo  Antes de rodar iniciar.bat, confira o config.cfg
echo  (copie de config.cfg.example se ainda nao existir).
echo ============================================
pause
