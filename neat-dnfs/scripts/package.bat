@echo off
:: Builds the release archive for Windows out of an existing Release build tree:
:: the three executables and their vcpkg DLLs, plus the config/ and templates/
:: they resolve at startup (the "runtime" install component -- see
:: CMakeLists.txt). Run scripts\build.bat first.

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set BUILD_DIR=%PROJECT_ROOT%\build\x64-release
set PACKAGE_DIR=%PROJECT_ROOT%\package

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo ERROR: no build found at %BUILD_DIR%. Run scripts\build.bat first.
    exit /b 1
)

set VERSION=
for /f "tokens=2 delims==" %%v in ('findstr /b /c:"NEAT_DNFS_VERSION:STRING=" "%BUILD_DIR%\CMakeCache.txt"') do set VERSION=%%v
if "%VERSION%"=="" (
    echo ERROR: could not read NEAT_DNFS_VERSION from %BUILD_DIR%\CMakeCache.txt.
    exit /b 1
)

set NAME=neat-dnfs-%VERSION%-windows-x64
set STAGE=%PACKAGE_DIR%\%NAME%

if exist "%STAGE%" rmdir /s /q "%STAGE%"
if exist "%PACKAGE_DIR%\%NAME%.zip" del /q "%PACKAGE_DIR%\%NAME%.zip"
mkdir "%PACKAGE_DIR%" 2>nul

cmake --install "%BUILD_DIR%" --config Release --component runtime --prefix "%STAGE%"
if errorlevel 1 ( echo ERROR: install failed. & exit /b 1 )

:: Compress-Archive on the directory itself, so the zip holds a single
:: top-level folder rather than spilling bin\ and share\ into the extract target.
powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%' -DestinationPath '%PACKAGE_DIR%\%NAME%.zip'"
if errorlevel 1 ( echo ERROR: archive creation failed. & exit /b 1 )

echo.
echo Package: %PACKAGE_DIR%\%NAME%.zip
