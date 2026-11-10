"""Create or update the local environment for the PyTorch lessons."""

from pathlib import Path
import platform
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    if (
        sys.version_info[:2] != (3, 10)
        or sys.platform not in ('win32', 'linux')
        or platform.machine().lower() not in ('amd64', 'x86_64')
        or sys.maxsize <= 2**32
    ):
        raise SystemExit('Use 64-bit Python 3.10 on Windows or x86-64 Linux.')

    env = WORKSPACE / '.venv'
    python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.is_file():
        if env.exists():
            raise SystemExit('The existing .venv has no usable interpreter. Rename it and rerun the installer.')
        subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)

    version = subprocess.check_output(
        [str(python), '-c', 'import sys; print(sys.version_info[:2], sys.maxsize > 2**32)'],
        text=True,
    ).strip()
    if version != '(3, 10) True':
        raise SystemExit('The existing .venv needs 64-bit Python 3.10. Rename it and rerun the installer.')

    pip = [str(python), '-m', 'pip', '--require-virtualenv', '--disable-pip-version-check']
    subprocess.run([*pip, 'install', '-r', str(WORKSPACE / 'pytorch/requirements.txt')], check=True)
    subprocess.run([
        *pip, 'install', '--index-url', 'https://download.pytorch.org/whl/cpu',
        'torch==2.4.1+cpu', 'torchvision==0.19.1+cpu',
    ], check=True)
    subprocess.run([*pip, 'check'], check=True)
    print('Installation complete. Open 01-blank-canvas or one of the preset launchers.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
