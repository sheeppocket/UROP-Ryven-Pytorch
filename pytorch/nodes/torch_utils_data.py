from dataclasses import dataclass

from ryven.node_env import *
from torch.utils.data import DataLoader

from .utils import value, boolean, clear_outputs

@dataclass(frozen=True)
class RemoteCsvDataset:
    path: str


@dataclass(frozen=True)
class RemoteImageFolder:
    path: str


@dataclass(frozen=True)
class RemoteFashionMNIST:
    root: str


@dataclass(frozen=True)
class RemoteBatches:
    dataset: RemoteCsvDataset | RemoteImageFolder | RemoteFashionMNIST
    batch_size: int
    shuffle: bool


# Earlier saved graphs may contain serialized CSV batch settings under this name.
RemoteCsvBatches = RemoteBatches


class RemoteCsvDatasetNode(Node):
    """Specify a CSV path on the training machine. Ryven does not open the file."""

    title = 'CSV File'
    init_inputs = [NodeInputType(label='csv_path', default=Data('train.csv'))]
    init_outputs = [NodeOutputType(label='dataset')]

    def place_event(self):
        self.update()

    def update_event(self, inp=-1):
        path = value(self, 0)
        dataset = RemoteCsvDataset(path.strip()) if isinstance(path, str) and path.strip() else None
        self.set_output_val(0, Data(dataset))


class RemoteImageFolderNode(Node):
    """Specify a folder with one subfolder per class on the training machine."""

    title = 'Image Folder'
    init_inputs = [NodeInputType(label='image_folder', default=Data('train'))]
    init_outputs = [NodeOutputType(label='dataset')]

    def place_event(self):
        self.update()

    def update_event(self, inp=-1):
        path = value(self, 0)
        dataset = RemoteImageFolder(path.strip()) if isinstance(path, str) and path.strip() else None
        self.set_output_val(0, Data(dataset))


class FashionMNISTNode(Node):
    """Download the training and test splits of Fashion-MNIST when the exported script runs."""

    title = 'Fashion-MNIST'
    init_inputs = [NodeInputType(label='data_root', default=Data('fashion-mnist-data'))]
    init_outputs = [NodeOutputType(label='dataset')]

    def place_event(self):
        self.update()

    def update_event(self, inp=-1):
        root = value(self, 0)
        dataset = RemoteFashionMNIST(root.strip()) if isinstance(root, str) and root.strip() else None
        self.set_output_val(0, Data(dataset))


class DataLoaderNode(Node):
    """Group a dataset into batches so the model can process several examples at once.
    Shuffling changes their order. Training and test datasets are grouped separately.
    """
    title = "DataLoader"
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

    def rebuilt(self):
        self.update()

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
            shuffle = boolean(value(self, 2, False))
            if isinstance(dataset, (RemoteCsvDataset, RemoteImageFolder, RemoteFashionMNIST)):
                self.set_output_val(0, Data(RemoteBatches(dataset, batch_size, shuffle)))
                return
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            self.set_output_val(0, Data(self.dataloader))
        except Exception as error:
            print(f'DataLoader: {error}')
