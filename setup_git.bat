@echo off
REM Quick Git PATH setup batch file
REM This adds Git to PATH for the current command prompt session

set "SCRIPT_DIR=%~dp0"
set "GIT_BIN=%SCRIPT_DIR%git-portable\bin"
set "GIT_CMD=%SCRIPT_DIR%git-portable\cmd"

if not exist "%GIT_BIN%\git.exe" (
    echo ERROR: Git not found at %GIT_BIN%\git.exe
    echo Please run install_git.ps1 first.
    exit /b 1
)

set "PATH=%PATH%;%GIT_BIN%;%GIT_CMD%"
echo Git added to PATH for this session.
echo.
git --version
echo.
echo You can now use 'git' command in this command prompt.
echo Note: This only works for the current command prompt window.

