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

if not exist "%IPK_INSTALL%\release" (
    if not exist "%IPK_SRC%" (
        echo Cloning imgui-platform-kit...
        git clone https://github.com/Jgocunha/imgui-platform-kit.git "%IPK_SRC%"
        if errorlevel 1 ( echo ERROR: Failed to clone imgui-platform-kit. & exit /b 1 )
        if defined IPK_REF (
            git -C "%IPK_SRC%" checkout %IPK_REF%
            if errorlevel 1 ( echo ERROR: Failed to checkout IPK_REF. & exit /b 1 )
        )
    )

    echo Building imgui-platform-kit Release...
    cmake -S "%IPK_SRC%\imgui-platform-kit" -B "%IPK_SRC%\build-release" ^
        -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
        -DCMAKE_INSTALL_PREFIX="%IPK_INSTALL%\release"
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release configure failed. & exit /b 1 )
    cmake --build "%IPK_SRC%\build-release" --config Release --parallel
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release build failed. & exit /b 1 )
    cmake --install "%IPK_SRC%\build-release" --config Release
    if errorlevel 1 ( echo ERROR: imgui-platform-kit Release install failed. & exit /b 1 )
) else (
    echo imgui-platform-kit Release already installed, skipping.
)

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping imgui-platform-kit Debug.
) else (
    if not exist "%IPK_INSTALL%\debug" (
        if not exist "%IPK_SRC%" (
            echo Cloning imgui-platform-kit...
            git clone https://github.com/Jgocunha/imgui-platform-kit.git "%IPK_SRC%"
            if errorlevel 1 ( echo ERROR: Failed to clone imgui-platform-kit. & exit /b 1 )
            if defined IPK_REF (
                git -C "%IPK_SRC%" checkout %IPK_REF%
                if errorlevel 1 ( echo ERROR: Failed to checkout IPK_REF. & exit /b 1 )
            )
        )

        echo Building imgui-platform-kit Debug...
        cmake -S "%IPK_SRC%\imgui-platform-kit" -B "%IPK_SRC%\build-debug" ^
            -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
            -DCMAKE_INSTALL_PREFIX="%IPK_INSTALL%\debug"
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug configure failed. & exit /b 1 )
        cmake --build "%IPK_SRC%\build-debug" --config Debug --parallel
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug build failed. & exit /b 1 )
        cmake --install "%IPK_SRC%\build-debug" --config Debug
        if errorlevel 1 ( echo ERROR: imgui-platform-kit Debug install failed. & exit /b 1 )
    ) else (
        echo imgui-platform-kit Debug already installed, skipping.
    )
)

:: ── dynamic-neural-field-composer ─────────────────────────────────────────────
set DNFC_SRC=%DEPS_DIR%\dynamic-neural-field-composer
set DNFC_INSTALL=%DEPS_DIR%\dnfc-install

if not exist "%DNFC_INSTALL%\release" (
    if not exist "%DNFC_SRC%" (
        echo Cloning dynamic-neural-field-composer...
        git clone https://github.com/Jgocunha/dynamic-neural-field-composer.git "%DNFC_SRC%"
        if errorlevel 1 ( echo ERROR: Failed to clone dynamic-neural-field-composer. & exit /b 1 )
        if defined DNFC_REF (
            git -C "%DNFC_SRC%" checkout %DNFC_REF%
            if errorlevel 1 ( echo ERROR: Failed to checkout DNFC_REF. & exit /b 1 )
        )
    )

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
) else (
    echo dynamic-neural-field-composer Release already installed, skipping.
)

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping dynamic-neural-field-composer Debug.
) else (
    if not exist "%DNFC_INSTALL%\debug" (
        if not exist "%DNFC_SRC%" (
            echo Cloning dynamic-neural-field-composer...
            git clone https://github.com/Jgocunha/dynamic-neural-field-composer.git "%DNFC_SRC%"
            if errorlevel 1 ( echo ERROR: Failed to clone dynamic-neural-field-composer. & exit /b 1 )
            if defined DNFC_REF (
                git -C "%DNFC_SRC%" checkout %DNFC_REF%
                if errorlevel 1 ( echo ERROR: Failed to checkout DNFC_REF. & exit /b 1 )
            )
        )

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
    ) else (
        echo dynamic-neural-field-composer Debug already installed, skipping.
    )
)

echo.
echo Setup complete. Run scripts\build.bat to build the project.
