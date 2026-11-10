from ryven.node_env import *

class NumpyToTensor(Node):
    """
    Converts numpy array to torch tensor using torch.from_numpy.
    """
    title = "Numpy to Tensor"
    init_inputs = [
        NodeInputType(label="numpy_array")
    ]
    init_outputs = [
        NodeOutputType(label="tensor")
    ]

    def update_event(self):
        import torch
        import numpy as np

        data = self.input(0)

        if data is None:
            return

        try:
            # Handle pandas DataFrame
            if hasattr(data, 'values'):
                data = data.values

            # Convert to numpy if not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            # Ensure array is contiguous for torch.from_numpy
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            # Convert to torch tensor
            tensor = torch.from_numpy(data)
            self.set_output_val(0, tensor)
        except Exception as e:
            self.set_output_val(0, None)
            print(f"Error converting to tensor: {e}")
