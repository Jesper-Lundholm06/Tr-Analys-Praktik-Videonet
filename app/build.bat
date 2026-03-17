@echo off
REM ============================================
REM  Bygg PlankInspection till Windows-tjänst
REM  Kör från projektets rotmapp (web-mappen / app)
REM ============================================

echo.
echo === STEG 1: Installera CPU-only PyTorch ===
echo (Avinstallerar eventuell CUDA-torch först)
pip uninstall torch torchvision torchaudio -y 2>nul
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo FEL: Kunde inte installera torch CPU.
    pause
    exit /b 1
)

echo.
echo === Verifierar torch-installation ===
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); import os; lib=os.path.join(os.path.dirname(torch.__file__),'lib'); dlls=[f for f in os.listdir(lib) if f.endswith('.dll')]; print('DLLs i torch/lib:', dlls)"
if errorlevel 1 (
    echo FEL: Torch-verifiering misslyckades.
    pause
    exit /b 1
)

echo.
echo *** CUDA available borde vara False ovan ***
echo *** Om det star True, skapa ett nytt venv forst ***
echo.

echo.
echo === STEG 2: Installera ovriga beroenden ===
pip install -r requirements.txt
if errorlevel 1 (
    echo FEL: Kunde inte installera beroenden.
    pause
    exit /b 1
)

echo.
echo === STEG 3: Bygg med PyInstaller ===
pyinstaller build.spec --noconfirm
if errorlevel 1 (
    echo FEL: PyInstaller misslyckades.
    pause
    exit /b 1
)

echo.
echo === Verifierar build ===
echo DLLs i dist\PlankInspection\_internal\torch\lib\:
dir dist\PlankInspection\_internal\torch\lib\*.dll /b 2>nul
if errorlevel 1 (
    echo VARNING: Inga torch DLL-filer hittades i builden!
    echo Kontrollera build.spec.
)

echo.
echo === STEG 4: Skapa installer (valfritt) ===
echo Vill du skapa en installer med Inno Setup?
echo (Kraver att Inno Setup 6+ ar installerat)
echo.
set /p BUILD_INSTALLER="Skapa installer? (j/n): "
if /i "%BUILD_INSTALLER%"=="j" (
    where iscc >nul 2>nul
    if errorlevel 1 (
        echo VARNING: iscc hittades inte i PATH.
        echo Installera Inno Setup eller kompilera manuellt:
        echo   iscc installer.iss
    ) else (
        iscc installer.iss
        if errorlevel 1 (
            echo FEL: Inno Setup misslyckades.
        ) else (
            echo Installer skapad i: installer_output\PlankInspection_Setup.exe
        )
    )
)

echo.
echo === KLAR ===
echo Utdata finns i:  dist\PlankInspection\
echo.
echo Testa manuellt:
echo   cd dist\PlankInspection
echo   service.exe install
echo   service.exe start
echo   (oppna http://localhost:8014 i webblasaren)
echo   service.exe stop
echo   service.exe remove
echo.
pause