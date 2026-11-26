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

:: ── vcpkg baseline ────────────────────────────────────────────────────────────
:: Every vcpkg dependency, imgui-platform-kit included, is declared in vcpkg.json and
:: installed by CMake on the first configure, at the versions of the vcpkg commit pinned
:: there as builtin-baseline. vcpkg reads those versions from the files it has checked
:: out, so VCPKG_ROOT has to be a full clone at that commit or a newer one.
set "BASELINE="
for /f "tokens=2 delims=:, " %%B in ('findstr /c:"builtin-baseline" "%PROJECT_ROOT%\vcpkg.json"') do set "BASELINE=%%~B"
if not defined BASELINE ( echo ERROR: no builtin-baseline found in vcpkg.json. & exit /b 1 )

set "SHALLOW="
for /f %%S in ('git -C "%VCPKG_ROOT%" rev-parse --is-shallow-repository 2^>nul') do set "SHALLOW=%%S"
if "%SHALLOW%"=="true" (
    echo vcpkg at "%VCPKG_ROOT%" is a shallow clone, fetching its full history...
    git -C "%VCPKG_ROOT%" fetch --unshallow origin
    if errorlevel 1 (
        echo ERROR: could not fetch the history of "%VCPKG_ROOT%". VCPKG_ROOT must be a clone you can write to.
        exit /b 1
    )
)
git -C "%VCPKG_ROOT%" merge-base --is-ancestor %BASELINE% HEAD 2>nul
if errorlevel 1 (
    echo ERROR: vcpkg at "%VCPKG_ROOT%" is older than the pinned baseline %BASELINE%, or is not a git clone.
    echo VCPKG_ROOT must be a git clone of https://github.com/microsoft/vcpkg at that commit or newer.
    echo To move it to the pinned baseline:
    echo   git -C "%VCPKG_ROOT%" fetch origin ^&^& git -C "%VCPKG_ROOT%" checkout %BASELINE% ^&^& "%VCPKG_ROOT%\bootstrap-vcpkg.bat" -disableMetrics
    exit /b 1
)

:: ── dynamic-neural-field-composer ─────────────────────────────────────────────
:: Pinned by scripts\dnfc-ref.txt; DNFC_REF overrides it. It is built against this
:: project's vcpkg.json (VCPKG_MANIFEST_DIR), so the library and this project link the
:: same dependency versions. Each install is stamped with the revision and a hash of the
:: manifest, so changing either one rebuilds it.
set DNFC_SRC=%DEPS_DIR%\dynamic-neural-field-composer
set DNFC_INSTALL=%DEPS_DIR%\dnfc-install
set DNFC_URL=https://github.com/Jgocunha/dynamic-neural-field-composer.git
if not defined DNFC_REF for /f "usebackq delims=" %%R in ("%SCRIPT_DIR%dnfc-ref.txt") do set "DNFC_REF=%%R"
set "MANIFEST_HASH="
for /f %%H in ('type "%PROJECT_ROOT%\vcpkg.json" "%PROJECT_ROOT%\vcpkg-configuration.json" "%PROJECT_ROOT%\ports\imgui-platform-kit\*" 2^>nul ^| git hash-object --stdin') do set "MANIFEST_HASH=%%H"
set "DNFC_STAMP=%DNFC_REF% %MANIFEST_HASH%"

call :invalidate_if_stale "%DNFC_INSTALL%\release" "%DNFC_STAMP%"
if not exist "%DNFC_INSTALL%\release" (
    call :ensure_src "%DNFC_SRC%" "%DNFC_URL%" "%DNFC_REF%"
    if errorlevel 1 exit /b 1

    echo Building dynamic-neural-field-composer Release...
    if exist "%DNFC_SRC%\build-release" rmdir /s /q "%DNFC_SRC%\build-release"
    cmake -S "%DNFC_SRC%\dynamic-neural-field-composer" -B "%DNFC_SRC%\build-release" ^
        -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
        -DCMAKE_INSTALL_PREFIX="%DNFC_INSTALL%\release" ^
        -DVCPKG_MANIFEST_DIR="%PROJECT_ROOT%" ^
        -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Release ^
        -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release ^
        -DDNF_COMPOSER_BUILD_TESTS=OFF
    if errorlevel 1 ( echo ERROR: dnfc Release configure failed. & exit /b 1 )
    cmake --build "%DNFC_SRC%\build-release" --config Release --parallel
    if errorlevel 1 ( echo ERROR: dnfc Release build failed. & exit /b 1 )
    cmake --install "%DNFC_SRC%\build-release" --config Release
    if errorlevel 1 ( echo ERROR: dnfc Release install failed. & exit /b 1 )
    >"%DNFC_INSTALL%\release\.revision" echo %DNFC_STAMP%
) else (
    echo dynamic-neural-field-composer Release already installed, skipping.
)

if "%NEAT_DNFS_RELEASE_ONLY%"=="1" (
    echo NEAT_DNFS_RELEASE_ONLY=1, skipping dynamic-neural-field-composer Debug.
) else (
    call :invalidate_if_stale "%DNFC_INSTALL%\debug" "%DNFC_STAMP%"
    if not exist "%DNFC_INSTALL%\debug" (
        call :ensure_src "%DNFC_SRC%" "%DNFC_URL%" "%DNFC_REF%"
        if errorlevel 1 exit /b 1

        echo Building dynamic-neural-field-composer Debug...
        if exist "%DNFC_SRC%\build-debug" rmdir /s /q "%DNFC_SRC%\build-debug"
        cmake -S "%DNFC_SRC%\dynamic-neural-field-composer" -B "%DNFC_SRC%\build-debug" ^
            -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
            -DCMAKE_INSTALL_PREFIX="%DNFC_INSTALL%\debug" ^
            -DVCPKG_MANIFEST_DIR="%PROJECT_ROOT%" ^
            -DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Debug ^
            -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Debug ^
            -DDNF_COMPOSER_BUILD_TESTS=OFF
        if errorlevel 1 ( echo ERROR: dnfc Debug configure failed. & exit /b 1 )
        cmake --build "%DNFC_SRC%\build-debug" --config Debug --parallel
        if errorlevel 1 ( echo ERROR: dnfc Debug build failed. & exit /b 1 )
        cmake --install "%DNFC_SRC%\build-debug" --config Debug
        if errorlevel 1 ( echo ERROR: dnfc Debug install failed. & exit /b 1 )
        >"%DNFC_INSTALL%\debug\.revision" echo %DNFC_STAMP%
    ) else (
        echo dynamic-neural-field-composer Debug already installed, skipping.
    )
)

echo.
echo Setup complete. Run scripts\build.bat to build the project.
exit /b 0

:: Clones SRC from URL if missing, then checks out REF -- independent of whether
:: the clone just happened, so a source directory left over from a previous run
:: still ends up on the right revision. Fetches first when an older clone lacks
:: REF. Safe to call more than once for the same SRC (once per Release/Debug).
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
git -C "%SRC%" cat-file -e %REF% 2>nul || git -C "%SRC%" fetch origin
if defined REF (
    git -C "%SRC%" checkout %REF%
    if errorlevel 1 ( echo ERROR: Failed to checkout %REF% in %SRC%. & exit /b 1 )
)
endlocal
goto :eof

:: Removes DIR if it was built from anything other than STAMP, so the caller's
:: own "already installed" check rebuilds instead of silently reusing a stale
:: build.
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
