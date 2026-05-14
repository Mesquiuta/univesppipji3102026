@echo off
REM Roda pipeline completo e em seguida abre o frontend Streamlit
echo ==========================================
echo Etapa 1/2: rodando pipeline analitico
echo ==========================================
python main.py --pipeline all
if errorlevel 1 (
    echo Pipeline falhou. Abortando.
    exit /b 1
)
echo.
echo ==========================================
echo Etapa 2/2: abrindo frontend Streamlit
echo ==========================================
python -m streamlit run app/Home.py
