import argparse
from pathlib import Path

from ryven.main.Ryven import run


def open_workspace(qt_app=None, *, project=None):
    """Open an editable project or a blank graph with the PyTorch nodes available."""
    nodes = Path(__file__).resolve().with_name('nodes')
    project_args = (str(project),) if project is not None else ()
    return run(
        *project_args, qt_app=qt_app, use_sysargs=False, show_dialog=False, qt_api='pyside2',
        nodes=[] if project_args else [nodes],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open a PyTorch workspace in Ryven.')
    parser.add_argument('project', nargs='?', type=Path, help='An optional Ryven project file.')
    open_workspace(project=parser.parse_args().project)
