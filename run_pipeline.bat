@echo off
REM Executa o pipeline analitico completo
echo ==========================================
echo Executando pipeline completo (main.py --pipeline all)
echo ==========================================
python main.py --pipeline all
if errorlevel 1 (
    echo Pipeline falhou.
    exit /b 1
)
echo.
echo Pipeline completo concluido. Artefatos em outputs/, models/ e data/.
