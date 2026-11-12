from __future__ import annotations

from dataclasses import dataclass, replace

from ryven.node_env import Data, Node, NodeInputType, NodeOutputType
from ryvencore.Base import Event
from torch import nn

from .utils import boolean, value


ACTIVATIONS = ('None', 'ReLU', 'LeakyReLU', 'GELU', 'Tanh', 'Sigmoid', 'Softmax')
RESNET_FEATURES = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}
ENCODER_WEIGHTS = ('DEFAULT', 'random')


@dataclass(frozen=True)
class ImageEncoderSpec:
    architecture: str
    weights: str
    frozen: bool


@dataclass(frozen=True)
class DenseModelSpec:
    input_features: int
    layers: tuple[tuple[int, str], ...] = ()
    encoder: ImageEncoderSpec | None = None
    vae_encoder: DenseModelSpec | None = None

    def linear_layers(self):
        input_width = self.input_features
        for neurons, activation in self.layers:
            yield input_width, neurons, activation
            input_width = neurons

    def build_model(self):
        modules = []
        for inputs, neurons, activation in self.linear_layers():
            modules.append(nn.Linear(inputs, neurons))
            if activation != 'None':
                kwargs = {'dim': -1} if activation == 'Softmax' else {}
                modules.append(getattr(nn, activation)(**kwargs))
        return nn.Sequential(*modules)


def positive_width(width, name):
    if type(width) is not int or width < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return width


class DenseInputNode(Node):
    """Start a dense model by specifying the number of features in one example."""

    title = 'Dense Input'
    instructions = 'Set the number of input features.'
    init_inputs = [NodeInputType(label='features', default=Data(2))]
    init_outputs = [NodeOutputType(label='architecture')]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def place_event(self):
        self.update()

    def update_event(self, inp=-1):
        try:
            features = positive_width(value(self, 0), 'Features')
        except ValueError as error:
            self.set_output_val(0, Data(None))
            self.report_changed.emit(str(error))
            return
        self.set_output_val(0, Data(DenseModelSpec(features)))
        self.report_changed.emit(f'{features} input features')


class DenseLayerNode(Node):
    """Append a fully connected layer and an optional activation. Its input width
    comes from the connected architecture.
    """

    title = 'Dense Layer'
    instructions = 'Connect an input, encoder, latent node, or another Dense Layer.'
    init_inputs = [
        NodeInputType(label='previous'),
        NodeInputType(label='neurons', default=Data(8)),
        NodeInputType(label='activation', default=Data('ReLU')),
    ]
    init_outputs = [NodeOutputType(label='model')]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def rebuilt(self):
        self.update()

    def update_event(self, inp=-1):
        previous = value(self, 0)
        try:
            if not isinstance(previous, DenseModelSpec):
                raise ValueError(self.instructions)
            neurons = positive_width(value(self, 1), 'Neurons')
            activation = value(self, 2, 'None')
            if activation not in ACTIVATIONS:
                raise ValueError('Choose a supported activation.')
        except ValueError as error:
            self.set_output_val(0, Data(None))
            self.report_changed.emit(str(error))
            return
        inputs = previous.layers[-1][0] if previous.layers else previous.input_features
        spec = replace(previous, layers=(*previous.layers, (neurons, activation)))
        self.set_output_val(0, Data(spec))
        self.report_changed.emit(f'{inputs} → {neurons} · {activation}')


class ImageEncoderNode(Node):
    """Describe a ResNet encoder for an exported image classifier. The generated
    script loads the model and removes its original classifier.
    """

    title = 'Image Encoder for Code Export'
    instructions = 'Choose an encoder, then connect Dense Layer nodes.'
    tags = ['backbone', 'resnet', 'transfer learning']
    init_inputs = [
        NodeInputType(label='architecture', default=Data('resnet18')),
        NodeInputType(label='weights', default=Data('DEFAULT')),
        NodeInputType(label='freeze', default=Data(True)),
    ]
    init_outputs = [NodeOutputType(label='architecture')]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def place_event(self):
        self.update()

    def update_event(self, inp=-1):
        try:
            architecture = value(self, 0)
            if not isinstance(architecture, str) or architecture not in RESNET_FEATURES:
                raise ValueError('Choose a supported ResNet encoder.')
            weights = value(self, 1)
            if weights not in ENCODER_WEIGHTS:
                raise ValueError('Choose DEFAULT or random weights.')
            frozen = boolean(value(self, 2, True))
            if weights == 'random' and frozen:
                raise ValueError('Set freeze to False to train from random weights.')
        except ValueError as error:
            self.set_output_val(0, Data(None))
            self.report_changed.emit(str(error))
            return
        width = RESNET_FEATURES[architecture]
        encoder = ImageEncoderSpec(architecture, weights, frozen)
        self.set_output_val(0, Data(DenseModelSpec(width, encoder=encoder)))
        mode = 'frozen' if frozen else 'trainable'
        self.report_changed.emit(f'{width} features per image · {mode}')
