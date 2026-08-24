@echo off

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DEPS_DIR=%PROJECT_ROOT%\deps
set IPK_RELEASE=%DEPS_DIR%\ipk-install\release
set IPK_DEBUG=%DEPS_DIR%\ipk-install\debug
set DNFC_RELEASE=%DEPS_DIR%\dnfc-install\release
set DNFC_DEBUG=%DEPS_DIR%\dnfc-install\debug

if not defined VCPKG_ROOT (
    echo ERROR: The environment variable VCPKG_ROOT is not set.
    echo Run scripts\setup.bat first to install all dependencies automatically.
    exit /b 1
)

mkdir %PROJECT_ROOT%\build\x64-release 2>nul

:: Release
cmake ^
    -S "%PROJECT_ROOT%" -B "%PROJECT_ROOT%\build\x64-release" ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%IPK_RELEASE%;%DNFC_RELEASE%" ^
    -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Release ^
    -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release
if errorlevel 1 ( echo ERROR: CMake Release configure failed. & exit /b 1 )

cmake --build "%PROJECT_ROOT%\build\x64-release" --config Release --parallel
if errorlevel 1 ( echo ERROR: Release build failed. & exit /b 1 )

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping Debug build.
    goto :done
)

mkdir %PROJECT_ROOT%\build\x64-debug 2>nul

:: Debug
cmake ^
    -S "%PROJECT_ROOT%" -B "%PROJECT_ROOT%\build\x64-debug" ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_PREFIX_PATH="%IPK_DEBUG%;%DNFC_DEBUG%" ^
    -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Debug ^
    -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Debug
if errorlevel 1 ( echo ERROR: CMake Debug configure failed. & exit /b 1 )

cmake --build "%PROJECT_ROOT%\build\x64-debug" --config Debug --parallel
if errorlevel 1 ( echo ERROR: Debug build failed. & exit /b 1 )

:done
echo.
echo Build complete.
