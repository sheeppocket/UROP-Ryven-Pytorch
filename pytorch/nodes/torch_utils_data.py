from ryven.node_env import *
from .utils import TransientData, value, boolean, clear_outputs
import torch
from torch.utils.data import DataLoader,  TensorDataset

class MakeTensorNode(Node):
    """Create a tensor, an array of numbers with a chosen shape, data type, and starting values."""

    title = "Create Tensor"
    init_inputs = [
        NodeInputType(label="shape"),
        NodeInputType(label="dtype", default=Data("float32")),
        NodeInputType(label="fill", default=Data("randn")),
    ]
    init_outputs = [
        NodeOutputType(label="tensor")
    ]

    def update_event(self, inp=-1):
        shape = value(self, 0)
        dtype = value(self, 1, "float32")
        fill = value(self, 2, "randn")

        if shape is None:
            clear_outputs(self)
            return

        try:
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
                raise ValueError('Shape must contain non-negative integers')
            aliases = {'float': 'float32', 'double': 'float64', 'half': 'float16',
                       'int': 'int32', 'long': 'int64'}
            if isinstance(dtype, str):
                dtype = getattr(torch, aliases.get(dtype.lower(), dtype.lower()), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError('Unknown tensor dtype')
            key = str(fill).lower()
            if key in ('zeros', 'zero', '0'):
                tensor = torch.zeros(shape, dtype=dtype)
            elif key in ('ones', 'one', '1'):
                tensor = torch.ones(shape, dtype=dtype)
            elif key in ('randn', 'normal', 'rand', 'uniform'):
                if not dtype.is_floating_point:
                    raise ValueError('Random fills require a floating point dtype; use a constant for labels')
                tensor = (torch.randn if key in ('randn', 'normal') else torch.rand)(shape, dtype=dtype)
            else:
                tensor = torch.full(shape, float(fill), dtype=dtype)
            self.set_output_val(0, Data(tensor))
        except Exception as error:
            clear_outputs(self)
            print(f'Make Tensor: {error}')


class TensorDatasetNode(Node):
    """Pair rows from tensors into examples. For classification, pair each image or
    feature vector with the label at the same position in the label tensor.
    """
    title = "Build Dataset from Tensors"
    tags = ['tensordataset']
    init_inputs = [
        # control input for GUI-only add-socket button (kept None by default)
        NodeInputType(label="", default=None),

        NodeInputType(label="tensor 1"),
        
    ]
    init_outputs = [
        NodeOutputType(label="dataset")
    ]

    def __init__(self, params):
        super().__init__(params)
        self.dataset = None
    
    def get_dataset(self):
        # Collect all non-None inputs; support inputs that are lists/tuples by flattening
        tensors = []
        for i in range(1, len(self.inputs)):
            val = value(self, i)
            if val is None:
                if self._inp_connected(i):
                    return None
                continue
            if isinstance(val, (list, tuple)):
                for v in val:
                    tensors.append(v)
            else:
                tensors.append(val)

        if not tensors:
            return None

        try:
            return TensorDataset(*tensors)
        except Exception as e:
            print(f"TensorDatasetNode: error creating TensorDataset: {e}")
            return None

    def update_event(self, inp=-1):
        self.dataset = self.get_dataset()
        image_tensor = self.dataset.tensors[0] if self.dataset is not None else None
        data_type = TransientData if hasattr(image_tensor, '_ryven_image_recipe') else Data
        self.set_output_val(0, data_type(self.dataset))

class DataLoaderNode(Node):
    """Group a dataset into batches so the model can process several examples at once.
    Shuffling changes their order. Training and test datasets are grouped separately.
    """
    title = "Group Examples into Batches"
    tags = ['dataloader', 'batch size']
    init_inputs = [
        NodeInputType(label="dataset"),
        NodeInputType(label="batch_size"),
        NodeInputType(label="shuffle")
    ]
    init_outputs = [
        NodeOutputType(label="dataloader")
    ]

    def __init__(self, params):
        super().__init__(params)
        self.dataloader = None

    def update_event(self, inp=-1):
        self.dataloader = None
        clear_outputs(self)
        try:
            data = self.input(0)
            dataset = data.payload if data is not None else None
            if dataset is None:
                return
            batch_size = int(value(self, 1, 32))
            if batch_size <= 0:
                raise ValueError('Batch size must be positive')
            self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                         shuffle=boolean(value(self, 2, False)))
            data_type = TransientData if isinstance(data, TransientData) else Data
            self.set_output_val(0, data_type(self.dataloader))
        except Exception as error:
            print(f'DataLoader: {error}')
