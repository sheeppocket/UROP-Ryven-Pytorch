import torch
import torch.nn as nn

from ryvencore import Node

class InputLayerNode(Node):
    title = 'Input Layer'
    init_inputs = [
        'shape',  # e.g., (1, 28, 28)
    ]
    init_outputs = [
        'layer',
    ]
    color = '#6c757d'

    def update_event(self, inp=-1):
        shape = self.input(0)
        # For PyTorch, input layer is just the shape info
        self.set_output_val(0, shape)

class HiddenLayerNode(Node):
    title = 'Hidden Layer'
    init_inputs = [
        'prev_layer',
        'layer_type',  # 'Dense', 'Conv2d', etc.
        'shape',       # output shape or units/filters
        'activation',  # activation function name
    ]
    init_outputs = [
        'layer',
    ]
    color = '#6c757d'

    def update_event(self, inp=-1):
        prev = self.input(0)
        layer_type = self.input(1)
        shape = self.input(2)
        activation = self.input(3)
        layer = None
        if layer_type == 'Dense':
            layer = nn.Linear(prev[-1], shape)
        elif layer_type == 'Conv2d':
            # prev: (batch, channels, H, W)
            in_channels = prev[1] if isinstance(prev, (tuple, list)) and len(prev) > 1 else 1
            layer = nn.Conv2d(in_channels, shape, 3, 1)
        # Add more types as needed
        # Attach activation if specified
        if activation:
            act = getattr(nn, activation, None)
            if act:
                layer = nn.Sequential(layer, act())
        self.set_output_val(0, layer)

class OutputLayerNode(Node):
    title = 'Output Layer'
    init_inputs = [
        'prev_layer',
        'shape',
        'activation',
    ]
    init_outputs = [
        'model',
    ]
    color = '#6c757d'

    def update_event(self, inp=-1):
        prev = self.input(0)
        shape = self.input(1)
        activation = self.input(2)
        # For demonstration, build a simple nn.Sequential model
        layers = []
        if isinstance(prev, nn.Module):
            layers.append(prev)
        elif isinstance(prev, (list, tuple)):
            layers.extend([l for l in prev if isinstance(l, nn.Module)])
        # Output layer (Dense)
        out_layer = nn.Linear(layers[-1].out_features if layers else shape, shape)
        if activation:
            act = getattr(nn, activation, None)
            if act:
                out_layer = nn.Sequential(out_layer, act())
        layers.append(out_layer)
        model = nn.Sequential(*layers)
        self.set_output_val(0, model)

class ExportModelNode(Node):
    title = 'Export Model'
    init_inputs = [
        'model',
        'file_path',
    ]
    init_outputs = [
        'file_path',
    ]
    color = '#6c757d'

    def update_event(self, inp=-1):
        model = self.input(0)
        file_path = self.input(1)
        if model and file_path:
            torch.save(model.state_dict(), file_path)
            self.set_output_val(0, file_path)
        else:
            self.set_output_val(0, None)
