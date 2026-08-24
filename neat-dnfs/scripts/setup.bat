@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DEPS_DIR=%PROJECT_ROOT%\deps

:: ── vcpkg ─────────────────────────────────────────────────────────────────────
if not defined VCPKG_ROOT (
    set VCPKG_ROOT=C:\tools\vcpkg
    echo VCPKG_ROOT not set. Installing vcpkg to !VCPKG_ROOT!...
    if not exist "!VCPKG_ROOT!" (
        git clone https://github.com/microsoft/vcpkg.git "!VCPKG_ROOT!"
        if errorlevel 1 ( echo ERROR: Failed to clone vcpkg. & exit /b 1 )
        call "!VCPKG_ROOT!\bootstrap-vcpkg.bat" -disableMetrics
        if errorlevel 1 ( echo ERROR: Failed to bootstrap vcpkg. & exit /b 1 )
    )
    setx VCPKG_ROOT "!VCPKG_ROOT!"
    echo VCPKG_ROOT set permanently to !VCPKG_ROOT!
    echo NOTE: Open a new terminal for VCPKG_ROOT to be visible to other tools.
)

:: ── vcpkg packages ────────────────────────────────────────────────────────────
echo Installing vcpkg packages...
"%VCPKG_ROOT%\vcpkg.exe" install ^
    "imgui[docking-experimental,core,opengl3-binding,glfw-binding,dx12-binding,win32-binding]:x64-windows" ^
    "implot:x64-windows" ^
    "imgui-node-editor:x64-windows" ^
    "nlohmann-json:x64-windows" ^
    "catch2:x64-windows" ^
    "fftw3:x64-windows"
if errorlevel 1 ( echo ERROR: vcpkg install failed. & exit /b 1 )

:: ── imgui-platform-kit ────────────────────────────────────────────────────────
set IPK_SRC=%DEPS_DIR%\imgui-platform-kit
set IPK_INSTALL=%DEPS_DIR%\ipk-install
set IPK_URL=https://github.com/Jgocunha/imgui-platform-kit.git

call :invalidate_if_stale "%IPK_INSTALL%\release" "%IPK_REF%"
if not exist "%IPK_INSTALL%\release" (
    call :ensure_src "%IPK_SRC%" "%IPK_URL%" "%IPK_REF%"
    if errorlevel 1 exit /b 1

    echo Building imgui-platform-kit Release...
    cmake -S "%IPK_SRC%\imgui-platform-kit" -B "%IPK_SRC%\build-release" ^
        -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
        -DCMAKE_INSTALL_PREFIX="%IPK_INSTALL%\release"
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release configure failed. & exit /b 1 )
    cmake --build "%IPK_SRC%\build-release" --config Release --parallel
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release build failed. & exit /b 1 )
    cmake --install "%IPK_SRC%\build-release" --config Release
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release install failed. & exit /b 1 )
    if defined IPK_REF ( >"%IPK_INSTALL%\release\.revision" echo %IPK_REF% )
) else (
    echo imgui-platform-kit Release already installed, skipping.
)

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping imgui-platform-kit Debug.
) else (
    call :invalidate_if_stale "%IPK_INSTALL%\debug" "%IPK_REF%"
    if not exist "%IPK_INSTALL%\debug" (
        call :ensure_src "%IPK_SRC%" "%IPK_URL%" "%IPK_REF%"
        if errorlevel 1 exit /b 1

        echo Building imgui-platform-kit Debug...
        cmake -S "%IPK_SRC%\imgui-platform-kit" -B "%IPK_SRC%\build-debug" ^
            -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
            -DCMAKE_INSTALL_PREFIX="%IPK_INSTALL%\debug"
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug configure failed. & exit /b 1 )
        cmake --build "%IPK_SRC%\build-debug" --config Debug --parallel
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug build failed. & exit /b 1 )
        cmake --install "%IPK_SRC%\build-debug" --config Debug
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug install failed. & exit /b 1 )
        if defined IPK_REF ( >"%IPK_INSTALL%\debug\.revision" echo %IPK_REF% )
    ) else (
        echo imgui-platform-kit Debug already installed, skipping.
    )
)

:: ── dynamic-neural-field-composer ─────────────────────────────────────────────
set DNFC_SRC=%DEPS_DIR%\dynamic-neural-field-composer
set DNFC_INSTALL=%DEPS_DIR%\dnfc-install
set DNFC_URL=https://github.com/Jgocunha/dynamic-neural-field-composer.git

call :invalidate_if_stale "%DNFC_INSTALL%\release" "%DNFC_REF%"
if not exist "%DNFC_INSTALL%\release" (
    call :ensure_src "%DNFC_SRC%" "%DNFC_URL%" "%DNFC_REF%"
    if errorlevel 1 exit /b 1

    echo Building dynamic-neural-field-composer Release...
    cmake -S "%DNFC_SRC%\dynamic-neural-field-composer" -B "%DNFC_SRC%\build-release" ^
        -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
        -DCMAKE_INSTALL_PREFIX="%DNFC_INSTALL%\release" ^
        -DCMAKE_PREFIX_PATH="%IPK_INSTALL%\release" ^
        -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Release ^
        -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release ^
        -DDNF_COMPOSER_BUILD_TESTS=OFF
    if errorlevel 1 ( echo ERROR: dnfc Release configure failed. & exit /b 1 )
    cmake --build "%DNFC_SRC%\build-release" --config Release --parallel
    if errorlevel 1 ( echo ERROR: dnfc Release build failed. & exit /b 1 )
    cmake --install "%DNFC_SRC%\build-release" --config Release
    if errorlevel 1 ( echo ERROR: dnfc Release install failed. & exit /b 1 )
    if defined DNFC_REF ( >"%DNFC_INSTALL%\release\.revision" echo %DNFC_REF% )
) else (
    echo dynamic-neural-field-composer Release already installed, skipping.
)

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping dynamic-neural-field-composer Debug.
) else (
    call :invalidate_if_stale "%DNFC_INSTALL%\debug" "%DNFC_REF%"
    if not exist "%DNFC_INSTALL%\debug" (
        call :ensure_src "%DNFC_SRC%" "%DNFC_URL%" "%DNFC_REF%"
        if errorlevel 1 exit /b 1

        echo Building dynamic-neural-field-composer Debug...
        cmake -S "%DNFC_SRC%\dynamic-neural-field-composer" -B "%DNFC_SRC%\build-debug" ^
            -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
            -DCMAKE_INSTALL_PREFIX="%DNFC_INSTALL%\debug" ^
            -DCMAKE_PREFIX_PATH="%IPK_INSTALL%\debug" ^
            -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Debug ^
            -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Debug ^
            -DDNF_COMPOSER_BUILD_TESTS=OFF
        if errorlevel 1 ( echo ERROR: dnfc Debug configure failed. & exit /b 1 )
        cmake --build "%DNFC_SRC%\build-debug" --config Debug --parallel
        if errorlevel 1 ( echo ERROR: dnfc Debug build failed. & exit /b 1 )
        cmake --install "%DNFC_SRC%\build-debug" --config Debug
        if errorlevel 1 ( echo ERROR: dnfc Debug install failed. & exit /b 1 )
        if defined DNFC_REF ( >"%DNFC_INSTALL%\debug\.revision" echo %DNFC_REF% )
    ) else (
        echo dynamic-neural-field-composer Debug already installed, skipping.
    )
)

echo.
echo Setup complete. Run scripts\build.bat to build the project.
exit /b 0

:: Clones SRC from URL if missing, then checks out REF if one was requested --
:: independent of whether the clone just happened, so a source directory left
:: over from a previous run still ends up on the right revision. Safe to call
:: more than once for the same SRC (e.g. once per Release/Debug build).
:ensure_src
setlocal
set "SRC=%~1"
set "URL=%~2"
set "REF=%~3"
if not exist "%SRC%" (
    echo Cloning %SRC%...
    git clone "%URL%" "%SRC%"
    if errorlevel 1 ( echo ERROR: Failed to clone %URL%. & exit /b 1 )
)
if defined REF (
    git -C "%SRC%" checkout %REF%
    if errorlevel 1 ( echo ERROR: Failed to checkout %REF% in %SRC%. & exit /b 1 )
)
endlocal
goto :eof

:: Removes DIR if it was built from a different revision than REF, so the
:: caller's own "already installed" check rebuilds instead of silently
:: reusing a stale revision. No-op when REF is unset (local dev default).
:invalidate_if_stale
setlocal
set "DIR=%~1"
set "REF=%~2"
if defined REF if exist "%DIR%" (
    set "REVISION="
    if exist "%DIR%\.revision" set /p REVISION=<"%DIR%\.revision"
    if not "!REVISION!"=="%REF%" (
        echo %DIR% was built from a different revision, rebuilding.
        rmdir /s /q "%DIR%"
    )
)
endlocal
goto :eof
