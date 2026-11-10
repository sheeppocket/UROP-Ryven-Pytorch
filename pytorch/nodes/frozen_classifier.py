from copy import deepcopy

from ryven.node_env import *
from ryvencore.Base import Event
import torch
from torch import nn

from .utils import clear_outputs, value


def find_classifier(model):
    if isinstance(model, nn.Module):
        for name in ('fc', 'classifier'):
            head = getattr(model, name, None)
            if isinstance(head, nn.Sequential) and len(head):
                head = head[-1]
            if isinstance(head, nn.Linear):
                return head
    raise ValueError('Connect a vision model with a final linear fc or classifier layer')


class FrozenVisionClassifier(nn.Module):
    """Keep a copy of the pretrained encoder in inference mode while fitting a new head."""

    def __init__(self, model, class_names):
        super().__init__()
        source_head = find_classifier(model)
        self.encoder = deepcopy(model, {id(source_head): nn.Identity()})
        self.encoder.requires_grad_(False).eval()
        self.classifier = nn.Linear(source_head.in_features, len(class_names),
                                    device=source_head.weight.device, dtype=source_head.weight.dtype)
        self._ryven_categories = tuple(class_names)
        self._ryven_model_spec = getattr(model, '_ryven_model_spec', None)

    @property
    def _ryven_transform(self):
        return self.encoder._ryven_transform

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != self.classifier.in_features:
            raise ValueError('The backbone must return one feature vector per image before its final classifier')
        return self.classifier(features)


# Older projects can contain a serialized instance under this name.
FrozenResNetClassifier = FrozenVisionClassifier


class FrozenResNetClassifierNode(Node):
    """Connect a vision model and the class names. This keeps its backbone fixed and
    replaces the final linear layer with a new trainable classifier. The backbone
    turns images into features. The classifier gives one score per class.
    """

    title = 'Add Trainable Classifier'
    tags = ['classifier head', 'fixed backbone', 'transfer learning']
    instructions = 'Connect a vision model and class names.'
    init_inputs = [NodeInputType(label='backbone'), NodeInputType(label='class_names'),
                   NodeInputType(label='trained_weights')]
    init_outputs = [NodeOutputType(label='model')]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def load_additional_data(self, data):
        if self.inputs and self.inputs[0].label_str == 'resnet':
            self.rename_input(0, 'backbone')
        if len(self.inputs) == 2:
            self.create_input(label='trained_weights')

    def update_event(self, inp=-1):
        model, names = value(self, 0), value(self, 1)
        if model is None or names is None:
            clear_outputs(self)
            self.report_changed.emit(self.instructions)
            return
        try:
            if not isinstance(names, (tuple, list)) or len(names) < 2:
                raise ValueError('Provide at least two class names in label order')
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError('Class names must be nonempty strings')
            names = [name.strip() for name in names]
            if len(set(names)) != len(names):
                raise ValueError('Class names must be unique')
            classifier = FrozenVisionClassifier(model, names)
            checkpoint_path = value(self, 2)
            if checkpoint_path:
                self.load_trained_weights(classifier, checkpoint_path)
        except Exception as error:
            print(f'Add Trainable Classifier: {error}')
            clear_outputs(self)
            self.report_changed.emit(str(error))
            return
        self.set_output_val(0, Data(classifier))
        head = classifier.classifier
        status = 'Trained classifier loaded' if checkpoint_path else 'Backbone stays fixed'
        self.report_changed.emit(f'{status}\n{head.in_features} features → {head.out_features} class scores')

    def load_trained_weights(self, classifier, path):
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not checkpoint.get('frozen_backbone'):
            raise ValueError('Choose a checkpoint from exported classifier-only training')
        spec = classifier._ryven_model_spec
        if spec is None or (checkpoint.get('architecture'), checkpoint.get('weights')) != spec[:2]:
            raise ValueError('Select the backbone architecture and weights used by the exported training code')
        if tuple(checkpoint.get('class_names', ())) != classifier._ryven_categories:
            raise ValueError('Use the same class names in the same order as the training checkpoint')
        if 'classifier_state_dict' in checkpoint:
            classifier.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        else:
            classifier.load_state_dict(checkpoint['model_state_dict'])
