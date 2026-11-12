from ryven.node_env import *

from .torch_utils_data import DataLoaderNode, FashionMNISTNode, RemoteCsvDatasetNode, RemoteImageFolderNode
from .dense_layers import DenseInputNode, DenseLayerNode, ImageEncoderNode
from .train_model import TrainModelNode
from .test_model import TestModelNode
from .learning_samples import SampleDatasetNode
from .vae import TrainVAENode, VariationalLatentNode

def export_group(category, node_types):
    export_nodes(node_types, sub_pkg_name=category)
    for node_type in node_types:
        # Ryven 3.5 prepends the prefix each time a Session registers a node type.
        # Store the full ID once so it stays stable across sessions.
        node_type.identifier = f'{node_type.identifier_prefix}.{node_type.__name__}'
        node_type.identifier_prefix = None


export_group('Data', [
    DataLoaderNode,
    FashionMNISTNode,
    SampleDatasetNode,
    RemoteCsvDatasetNode,
    RemoteImageFolderNode,
])

export_group('Model Setup', [
    DenseInputNode,
    DenseLayerNode,
    ImageEncoderNode,
    VariationalLatentNode,
])

export_group('Training', [TrainModelNode, TrainVAENode])
export_group('Evaluation', [TestModelNode])

@on_gui_load
def load_gui():
    from . import gui
