"""Shared value handling for the Ryven 3.5 / ryvencore 0.5 API."""
from ryvencore import Data


class TransientData(Data):
    """Save as empty Data so the graph can rebuild derived values when reopened."""

    def get_data(self):
        return None


def value(node, index, default=None):
    data = node.input(index)
    result = data.payload if isinstance(data, Data) else data
    return default if result is None else result


def boolean(value):
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes', 'y'):
            return True
        if value.lower() in ('false', '0', 'no', 'n', ''):
            return False
        raise ValueError(f'Invalid boolean: {value}')
    return bool(value)


def clear_outputs(node):
    for index in range(len(node.outputs)):
        node.set_output_val(index, Data(None))
