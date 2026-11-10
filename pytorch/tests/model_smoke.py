"""Offline architecture/weight checks. Processes models sequentially to limit memory."""
import os
os.environ['RYVEN_MODE'] = 'no-gui'
import gc
import importlib
from pathlib import Path
from unittest.mock import patch
import torch
from ryven.main.packages.nodes_package import import_nodes_package
classes, _ = import_nodes_package(directory=str(Path(__file__).resolve().parents[1] / 'nodes'))
cls = next(c for c in classes if c.__name__ == 'ModelsNode')
package = cls.__module__.rsplit('.', 1)[0]
FrozenVisionClassifier = importlib.import_module(package + '.frozen_classifier').FrozenVisionClassifier
train_model = importlib.import_module(package + '.train_model').train_model
# get_model does not need a graph instance.
node = object.__new__(cls)
torch.set_num_threads(2)
for name in cls.MODEL_OPTIONS:
    weights = 'IMAGENET1K_V1' if name == 'mobilenet_v2' else 'DEFAULT'
    with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Unexpected download')):
        model = node.get_model(name, weights)
        model.eval()
        images = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            result = model(images)
        assert result.shape == (1, 1000), (name, result.shape)
        assert torch.isfinite(result).all(), name
        assert callable(model._ryven_transform)
        classifier = FrozenVisionClassifier(model, ('first', 'second'))
        before = classifier.classifier.bias.detach().clone()
        assert [key for key, p in classifier.named_parameters() if p.requires_grad] == [
            'classifier.weight', 'classifier.bias',
        ]
        dataset = torch.utils.data.TensorDataset(images, torch.tensor([1]))
        train_model(classifier, torch.utils.data.DataLoader(dataset))
        assert not torch.equal(before, classifier.classifier.bias), name
        assert all(not module.training for module in classifier.encoder.modules()), name
        stock_state = model.state_dict()
        assert all(torch.equal(stock_state[key], tensor) for key, tensor in classifier.encoder.state_dict().items()), name
        with torch.no_grad():
            adapted = classifier(images)
        assert adapted.shape == (1, 2) and torch.isfinite(adapted).all(), name
        print(name, weights, 'stock output + classifier-only training PASS', flush=True)
    del model, classifier, stock_state
    gc.collect()
print('ALL SIX BUNDLED MODELS PASSED', flush=True)
