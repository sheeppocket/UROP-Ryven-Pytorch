
from ryven.node_env import *
import torch
import torchvision.models as models
from pathlib import Path
from .utils import value

# Place .pth weight files here to avoid downloading
WEIGHTS_DIR = Path(__file__).parent / 'pretrained_weights'

MODEL_CONSTRUCTORS = {
    'resnet18':     models.resnet18,
    'resnet34':     models.resnet34,
    'resnet50':     models.resnet50,
    'mobilenet_v2': models.mobilenet_v2,
    'vgg13':        models.vgg13,
    'vgg16':        models.vgg16,
}


class ModelsNode(Node):
    """Choose a vision model and load its weights. A pretrained model starts with patterns
    learned from another dataset. Connect it to Add Trainable Classifier to reuse the
    backbone for your own classes.
    """

    title = "Load Vision Model"
    tags = ['backbone', 'pretrained model', 'resnet']
    init_inputs = [
        NodeInputType(label="model_type", default=Data("resnet18")),
        NodeInputType(label="weights", default=None),
    ]
    init_outputs = [
        NodeOutputType(label="model"),
    ]

    MODEL_OPTIONS = [
        "mobilenet_v2",
        "resnet18",
        "resnet34",
        "resnet50",
        "vgg13",
        "vgg16",
    ]

    def __init__(self, params):
        super().__init__(params)
        self.model = None

    def get_model(self, model_type, weights=None):
        model_type = str(model_type or 'resnet18').lower()
        if model_type not in MODEL_CONSTRUCTORS:
            raise ValueError(f'Unknown model type: {model_type}')
        constructor = MODEL_CONSTRUCTORS[model_type]
        selection = str(weights).lower() if weights is not None else 'default'
        if selection in ('random', 'none'):
            result = constructor(weights=None)
            result._ryven_model_spec = (model_type, None, None)
            return result
        enum = models.get_model_weights(constructor)
        if isinstance(weights, enum):
            selected = weights
        elif weights is None or selection in ('', 'default'):
            selected = enum.DEFAULT
        elif Path(str(weights)).is_file():
            checkpoint = Path(str(weights)).resolve()
            result = constructor(weights=None)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            result.load_state_dict(state.get('model_state_dict', state))
            if state.get('architecture') == model_type and state.get('frozen_backbone') is False and state.get('weights'):
                selected = enum.verify(state['weights'])
                result._ryven_transform = selected.transforms()
                result._ryven_categories = selected.meta.get('categories')
            result._ryven_model_spec = (model_type, None, str(checkpoint))
            return result
        else:
            selected = enum.verify(weights)
        filename = selected.url.rsplit('/', 1)[-1]
        local_file = WEIGHTS_DIR / filename
        if local_file.exists():
            result = constructor(weights=None)
            result.load_state_dict(torch.load(str(local_file), map_location='cpu', weights_only=True))
        else:
            result = constructor(weights=selected, progress=False)
        result._ryven_transform = selected.transforms()
        result._ryven_categories = selected.meta.get('categories')
        result._ryven_model_spec = (model_type, selected.name, None)
        return result

    def update_event(self, inp=-1):
        # Input edits invalidate the old model; the Load Model button loads it.
        self.model = None
        self.set_output_val(0, Data(None))
        if inp >= 0:
            return
        try:
            self.model = self.get_model(value(self, 0, 'resnet18'), value(self, 1))
            self.set_output_val(0, Data(self.model))
        except Exception as error:
            print(f'Model: {error}')
