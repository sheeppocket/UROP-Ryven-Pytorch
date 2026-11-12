$ErrorActionPreference = 'Stop'

function Test-Python310($executable, $interpreterArgs = @()) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        return $false
    }

    $ErrorActionPreference = 'SilentlyContinue'
    $probe = & $executable @interpreterArgs -c 'import platform, sys; print(sys.version_info[:2], sys.maxsize > 2**32, platform.machine().lower())' 2>$null
    return ($LASTEXITCODE -eq 0 -and $probe -in '(3, 10) True amd64', '(3, 10) True x86_64')
}

function Install-Python310($targetDir) {
    # The SHA-256 is from Microsoft's WinGet manifest for Python.Python.3.10, x64, version 3.10.11.
    $installerUrl = 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe'
    $expectedHash = 'D8DEDE5005564B408BA50317108B765ED9C3C510342A598F9FD42681CBE0648B'
    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ('urop-python-' + [guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $tempDir
    $installer = Join-Path $tempDir 'python-3.10.11-amd64.exe'

    try {
        Write-Host 'Downloading Python 3.10.11 (64-bit) from python.org...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $installer -TimeoutSec 120

        if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $expectedHash) {
            throw 'The downloaded Python installer failed SHA-256 verification.'
        }

        Write-Host 'Installing Python for the current Windows user...'
        $installerArgs = @(
            '/quiet', '/norestart', 'InstallAllUsers=0', 'Include_launcher=0',
            'PrependPath=0', 'Shortcuts=0', 'Include_pip=1', 'Include_test=0',
            ('TargetDir="{0}"' -f $targetDir)
        )
        $process = Start-Process -FilePath $installer -ArgumentList $installerArgs -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "The Python installer exited with code $($process.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
        if (-not (Get-ChildItem -LiteralPath $tempDir -Force)) {
            Remove-Item -LiteralPath $tempDir -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\..')
    $pythonArgs = @()
    if (Test-Path -LiteralPath '.venv') {
        $python = (Join-Path (Get-Location).Path '.venv\Scripts\python.exe')
        if (-not (Test-Python310 $python)) {
            throw 'The existing .venv needs a working 64-bit Python 3.10. Rename .venv and rerun the installer.'
        }
    } else {
        $targetDir = Join-Path $env:LOCALAPPDATA 'UROP\Python310'
        $managedPython = Join-Path $targetDir 'python.exe'
        $userPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
        $python = @($managedPython, $userPython) | Where-Object { Test-Python310 $_ } | Select-Object -First 1
        if (-not $python) {
            # Only look for a runtime already installed when probing either Python launcher.
            $env:PYTHON_MANAGER_AUTOMATIC_INSTALL = 'false'
            Remove-Item Env:PYLAUNCHER_ALLOW_INSTALL, Env:PYLAUNCHER_ALWAYS_INSTALL -ErrorAction SilentlyContinue
            $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($launcher -and (Test-Python310 $launcher.Source @('-3.10'))) {
                $python = $launcher.Source
                $pythonArgs = @('-3.10')
            } else {
                if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
                    throw 'This installer requires x64 Windows.'
                }
                Install-Python310 $targetDir
                if (-not (Test-Python310 $managedPython)) {
                    throw 'The installed Python 3.10 interpreter could not be started.'
                }
                $python = $managedPython
            }
        }
    }

    & $python @pythonArgs 'launchers/install.py'
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
