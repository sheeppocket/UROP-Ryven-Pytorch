@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo Run 00-install.cmd first to create .venv. 1>&2
    popd
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "pytorch\open_ryven.py" %1
set "launcher_status=%errorlevel%"
popd
if not "%launcher_status%"=="0" pause
exit /b %launcher_status%
