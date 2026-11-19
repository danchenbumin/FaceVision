@echo off
setlocal

if "%~1"=="" (
    echo Usage: deploy_windows.bat ^<path-to-FaceVision.exe^>
    exit /b 1
)

set "APP=%~f1"
if not exist "%APP%" (
    echo Executable not found: %APP%
    exit /b 2
)

where windeployqt >nul 2>nul
if errorlevel 1 (
    echo windeployqt is not in PATH. Open a Qt command prompt or add the Qt bin directory to PATH.
    exit /b 3
)

windeployqt --no-translations --compiler-runtime "%APP%"
if errorlevel 1 exit /b 4

echo Deployment completed.
endlocal
