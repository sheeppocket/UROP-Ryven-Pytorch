@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "launchers\install.py"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "launchers\install.py"
) else (
    py -3.10 --version >nul 2>&1
    if errorlevel 1 (
        echo Install 64-bit Python 3.10 with the Python Launcher, then run this installer again. 1>&2
        popd
        pause
        exit /b 1
    )
    py -3.10 "launchers\install.py"
)
set "install_status=%errorlevel%"
popd
pause
exit /b %install_status%
