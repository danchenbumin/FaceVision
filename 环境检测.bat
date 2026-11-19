@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "OUT=%~dp0environment_report.txt"
(
    echo ================================================
    echo FaceVision Windows 环境检测
    echo 时间: %date% %time%
    echo ================================================
    echo.

    echo [系统]
    ver
    echo PROCESSOR_ARCHITECTURE=%PROCESSOR_ARCHITECTURE%
    echo.

    echo [CMake]
    where cmake 2^>nul
    cmake --version 2^>nul
    echo.

    echo [Ninja]
    where ninja 2^>nul
    ninja --version 2^>nul
    echo.

    echo [MSVC]
    where cl 2^>nul
    cl 2^>^&1
    echo.

    echo [MinGW]
    where g++ 2^>nul
    g++ --version 2^>nul
    echo.

    echo [Qt tools]
    where qmake 2^>nul
    qmake -v 2^>nul
    where qtpaths6 2^>nul
    qtpaths6 --qt-version 2^>nul
    where windeployqt 2^>nul
    echo.

    echo [Qt common directories]
    if exist C:\Qt dir /b C:\Qt
    if exist D:\Qt dir /b D:\Qt
    echo.

    echo [OpenCV environment variables]
    echo OpenCV_DIR=%OpenCV_DIR%
    echo OpenCV_ROOT=%OpenCV_ROOT%
    echo.

    echo [OpenCVConfig.cmake quick search]
    for %%D in (C:\opencv D:\opencv C:\openCV D:\openCV C:\libs D:\libs) do (
        if exist "%%D" (
            for /r "%%D" %%F in (OpenCVConfig.cmake) do echo %%F
        )
    )
    echo.

    echo [Visual Studio]
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
        "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -all -products * -property installationPath
    ) else (
        echo vswhere.exe not found
    )
) > "%OUT%" 2>&1

type "%OUT%"
echo.
echo 检测结果已保存到:
echo %OUT%
pause
endlocal
