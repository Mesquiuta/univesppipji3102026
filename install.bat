@echo off
REM Instala dependências do projeto
echo ==========================================
echo Instalando dependencias do projeto
echo ==========================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Falha ao instalar dependencias.
    exit /b 1
)
echo.
echo Dependencias instaladas com sucesso.
