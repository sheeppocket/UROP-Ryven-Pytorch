@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0_install.ps1"
set "install_status=%errorlevel%"
pause
exit /b %install_status%
