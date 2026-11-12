from ryven.node_env import *

from .torch_utils_data import (
    MakeTensorNode, TensorDatasetNode, DataLoaderNode, RemoteCsvDatasetNode, RemoteImageFolderNode,
)
from .dense_layers import DenseInputNode, DenseLayerNode, ImageEncoderNode
from .models import ModelsNode
from .train_model import TrainModelNode
from .weights import PresetWeightsNode
from .test_model import TestModelNode
from .image_nodes import ImagesToTensorNode, EvaluateModelNode
from .frozen_classifier import FrozenResNetClassifierNode
from .digits_demo import DigitsDemoNode
from .image_folder_split import ImageFolderPathsNode, ImageFolderSplitNode
from .learning_samples import SampleDatasetNode, SampleModelNode

def export_group(category, node_types):
    export_nodes(node_types, sub_pkg_name=category)
    for node_type in node_types:
        # Ryven 3.5 prepends the prefix each time a Session registers a node type.
        # Store the full ID once so it stays stable across sessions.
        node_type.identifier = f'{node_type.identifier_prefix}.{node_type.__name__}'
        node_type.identifier_prefix = None


export_group('Data', [
    MakeTensorNode,
    TensorDatasetNode,
    DataLoaderNode,
    ImagesToTensorNode,
    ImageFolderPathsNode,
    ImageFolderSplitNode,
    SampleDatasetNode,
    RemoteCsvDatasetNode,
    RemoteImageFolderNode,
])

export_group('Model Setup', [
    ModelsNode,
    PresetWeightsNode,
    FrozenResNetClassifierNode,
    SampleModelNode,
    DenseInputNode,
    DenseLayerNode,
    ImageEncoderNode,
])

export_group('Training', [TrainModelNode])
export_group('Evaluation', [TestModelNode, EvaluateModelNode])
export_group('Legacy', [DigitsDemoNode])

@on_gui_load
def load_gui():
    from . import gui
