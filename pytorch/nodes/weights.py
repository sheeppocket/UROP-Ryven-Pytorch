from ryven.node_env import *
from .utils import value

class CustomWeightsNode(Node):
    """Outputs a file path string for weights."""
    title = "Custom Weights"
    init_inputs = [
        NodeInputType(label="file_path", default=Data("")),
    ]
    init_outputs = [
        NodeOutputType(label="weights"),
    ]
    def update_event(self, inp=-1):
        file_path = value(self, 0)
        self.set_output_val(0, Data(file_path))


class PresetWeightsNode(Node):
    """Select the default pretrained weights for a vision model. Connect this to the
    model loader's weights input to start from weights learned on ImageNet.
    """

    title = "Use Default Pretrained Weights"
    MODEL_OPTIONS = [
        "mobilenet_v2",
        "resnet18",
        "resnet34",
        "resnet50",
        "vgg13",
        "vgg16",
    ]
    init_inputs = [
        NodeInputType(label="model_type", default=Data("resnet18")),
    ]
    init_outputs = [
        NodeOutputType(label="weights"),
    ]
    def update_event(self, inp=-1):
        model_type = value(self, 0)
        if model_type is None:
            self.set_output_val(0, Data(None))
            return
        # Resolve the architecture-specific default in the Model node
        self.set_output_val(0, Data("DEFAULT"))
