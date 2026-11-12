"""Integration tests using real Ryven nodes, ports and flow propagation."""
import os
os.environ['RYVEN_MODE'] = 'no-gui'
from contextlib import redirect_stdout
import importlib
from io import StringIO
import json
from pathlib import Path
from struct import pack
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from ryvencore import Data, Session
from ryven.main.packages.nodes_package import import_nodes_package

ROOT = Path(__file__).resolve().parents[1]
CLASSES, _ = import_nodes_package(directory=str(ROOT / 'nodes'))
TYPES = {cls.__name__: cls for cls in CLASSES}
PACKAGE = TYPES['TrainModelNode'].__module__.rsplit('.', 1)[0]
training = importlib.import_module(PACKAGE + '.train_model')

class NodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.session = Session()
        self.session.register_node_types(CLASSES)
        self.flow = self.session.create_flow('test')
        self.errors = []

    def node(self, name, inputs=()):
        node = self.flow.create_node(TYPES[name])
        node.update_error.sub(self.errors.append)
        for index, item in enumerate(inputs):
            node.inputs[index].default = Data(item)
        return node

    def output(self, node, index=0):
        result = node.outputs[index].val
        self.assertIsInstance(result, Data)
        self.assertEqual(self.errors, [])
        return result.payload

    def test_all_exported_callbacks_and_serialization(self):
        for name in TYPES:
            node = self.node(name)
            node.update(inp=0)
        self.assertEqual(self.errors, [])
        saved = self.session.serialize()
        json.dumps(saved)
        other = Session()
        other.register_node_types(CLASSES)
        other.load(saved)
        self.assertEqual(len(other.flows[0].nodes), len(TYPES))

    def test_linear_preset_trains_and_evaluates(self):
        flow, = self.session.load(json.loads((ROOT / 'linear_demo.json').read_text(encoding='utf-8')))
        nodes = {type(node).__name__: node for node in flow.nodes}
        data = nodes['SampleDatasetNode']
        data.update()
        train_data, test_data = self.output(data, 0), self.output(data, 1)
        self.assertEqual(len(train_data), 128)
        self.assertEqual(len(test_data), 64)
        self.assertEqual(tuple(train_data[0][0].shape), (2,))
        self.assertFalse(torch.equal(train_data[0][0], test_data[0][0]))
        trainer = nodes['TrainModelNode']
        torch.manual_seed(42)
        history = trainer.train()
        self.assertEqual(len(history), 8)
        self.assertLess(history[-1], history[0])
        self.assertGreaterEqual(nodes['TestModelNode'].accuracy, .95)
        code = trainer.export_training_code()
        self.assertIn('nn.Linear(2,', code)
        namespace = {'__name__': 'sample_export'}
        exec(compile(code, '<sample export>', 'exec'), namespace)
        regenerated = namespace['make_training_dataset']()
        self.assertTrue(torch.equal(regenerated.tensors[0], train_data.tensors[0]))
        self.assertTrue(torch.equal(regenerated.tensors[1], train_data.tensors[1]))

    def test_fashion_mnist_vae_preset_exports_training_and_preview(self):
        flow, = self.session.load(json.loads((ROOT / 'fashion_mnist_vae.json').read_text(encoding='utf-8')))
        nodes = {type(node).__name__: node for node in flow.nodes}
        trainer = nodes['TrainVAENode']
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data_root = folder / 'dataset'
            data_node = nodes['FashionMNISTNode']
            data_node.inputs[0].default = Data(str(data_root))
            data_node.update(0)
            code = trainer.export_training_code()
            self.assertFalse(data_root.exists())
            self.assertIn('nn.Linear(784, 128)', code)
            self.assertIn('LATENT_DIM = 8', code)
            self.assertIn('nn.Linear(8, 128)', code)
            self.assertIn('nn.Linear(128, 784)', code)

            # Use the real torchvision loader with tiny, cached train and test files.
            raw = data_root / 'FashionMNIST' / 'raw'
            raw.mkdir(parents=True)
            for split, labels in (('train', [0, 1, 0, 1]), ('t10k', [1, 0])):
                pixels = b''.join(bytes([label * 255]) * 784 for label in labels)
                (raw / f'{split}-images-idx3-ubyte').write_bytes(pack('>IIII', 2051, len(labels), 28, 28) + pixels)
                (raw / f'{split}-labels-idx1-ubyte').write_bytes(pack('>II', 2049, len(labels)) + bytes(labels))

            script = {'__name__': 'fashion_vae_export'}
            exec(compile(code, '<Fashion-MNIST VAE export>', 'exec'), script)
            checkpoint_path = folder / 'vae.pth'
            preview_path = folder / 'preview.png'
            script['SETTINGS'].update(epochs=1, checkpoint=str(checkpoint_path), preview=str(preview_path))
            with redirect_stdout(StringIO()) as stdout:
                script['main']()
            self.assertIn('with 4 images', stdout.getvalue())
            self.assertIn('Test (2 images) | reconstruction=', stdout.getvalue())
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint['latent_dimensions'], 8)
            self.assertEqual(checkpoint['model_state_dict']['mean.weight'].shape, (8, 128))
            self.assertEqual(checkpoint['model_state_dict']['log_variance.weight'].shape, (8, 128))
            self.assertGreater(checkpoint['history'][0]['reconstruction'], 0)
            self.assertGreater(checkpoint['test']['kl'], 0)
            with Image.open(preview_path) as preview:
                self.assertEqual(preview.size, (62, 92))

    def test_dense_graph_exports_standalone_csv_training_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            csv_path = folder / 'training data.csv'
            features = self.node('DenseInputNode', [2])
            hidden = self.node('DenseLayerNode', [None, 8, 'Tanh'])
            output = self.node('DenseLayerNode', [None, 2, 'None'])
            remote_data = self.node('RemoteCsvDatasetNode', [str(csv_path)])
            loader = self.node('DataLoaderNode', [None, 2, True])
            trainer = self.node('TrainModelNode', [None, None, 'cross_entropy', 'AdamW', 0.01, 2])
            features.update()
            remote_data.update()
            self.flow.connect_nodes(features.outputs[0], hidden.inputs[0])
            self.flow.connect_nodes(hidden.outputs[0], output.inputs[0])
            self.flow.connect_nodes(output.outputs[0], trainer.inputs[0])
            self.flow.connect_nodes(remote_data.outputs[0], loader.inputs[0])
            self.flow.connect_nodes(loader.outputs[0], trainer.inputs[1])

            hidden.inputs[1].default = Data(5)
            hidden.update(1)
            code = trainer.export_training_code()
            self.assertIn('nn.Linear(2, 5)', code)
            self.assertIn('nn.Tanh()', code)
            self.assertIn('nn.Linear(5, 2)', code)
            self.assertFalse(csv_path.exists())
            with self.assertRaisesRegex(ValueError, 'Copy Training Code'):
                trainer.train()

            saved = json.loads(json.dumps(self.session.serialize()))
            session = Session()
            session.register_node_types(CLASSES)
            flow, = session.load(saved)
            restored = next(node for node in flow.nodes if isinstance(node, TYPES['TrainModelNode']))
            self.assertEqual(restored.export_training_code(), code)

            csv_path.write_text('x,y,label\n-1,0,0\n-1,1,0\n1,0,1\n1,1,1\n', encoding='utf-8')
            script_path = folder / 'train_dense.py'
            script_path.write_text(code, encoding='utf-8')
            result = subprocess.run([sys.executable, str(script_path)], cwd=folder,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Epoch 2/2', result.stdout)
            checkpoint = torch.load(folder / 'training_result.pth', weights_only=True)
            self.assertEqual(len(checkpoint['loss_history']), 2)
            self.assertEqual(checkpoint['model_state_dict']['0.weight'].shape, (5, 2))
            self.assertEqual(checkpoint['model_state_dict']['2.weight'].shape, (2, 5))

    def test_encoder_graph_exports_image_training_with_a_custom_head(self):
        for architecture, width, classifier in (('resnet18', 512, 'fc'), ('mobilenet_v2', 1280, 'classifier')):
            with self.subTest(architecture=architecture):
                self.check_encoder_export(architecture, width, classifier)

    def check_encoder_export(self, architecture, width, classifier):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image_folder = folder / 'train'
            encoder = self.node('ImageEncoderNode', [architecture, 'DEFAULT', True])
            hidden = self.node('DenseLayerNode', [None, 4, 'ReLU'])
            output = self.node('DenseLayerNode', [None, 2, 'None'])
            images = self.node('RemoteImageFolderNode', [str(image_folder)])
            batches = self.node('DataLoaderNode', [None, 2, True])
            trainer = self.node('TrainModelNode')
            for source, target, index in (
                (encoder, hidden, 0), (hidden, output, 0), (output, trainer, 0),
                (images, batches, 0), (batches, trainer, 1),
            ):
                self.flow.connect_nodes(source.outputs[0], target.inputs[index])
            with patch(f'torchvision.models.{architecture}', side_effect=AssertionError('Encoder loaded locally')):
                encoder.update()
                images.update()
                code = trainer.export_training_code()
            self.assertFalse(image_folder.exists())
            self.assertIn(f'nn.Linear({width}, 4)', code)
            self.assertIn('nn.Linear(4, 2)', code)
            self.assertIn(f'encoder.{classifier} = nn.Identity()', code)
            with self.assertRaisesRegex(ValueError, 'Copy Training Code'):
                trainer.train()

            for category, color in (('blue', 'blue'), ('red', 'red')):
                class_folder = image_folder / category
                class_folder.mkdir(parents=True)
                Image.new('RGB', (28, 28), color).save(class_folder / 'example.png')

            script = {'__name__': 'image_training_export'}
            exec(compile(code, '<image training export>', 'exec'), script)
            models = script['models']
            build_encoder = getattr(models, architecture)
            weights = models.get_model_weights(build_encoder).DEFAULT
            backbone = build_encoder(weights=None)
            batch_norm = next(module for module in backbone.modules() if isinstance(module, torch.nn.BatchNorm2d))
            bn_mean = batch_norm.running_mean.clone()
            # Exercise the frozen path without fetching weights in the test.
            with patch.object(models, architecture, return_value=backbone) as constructor:
                script['SETTINGS']['output_checkpoint'] = str(folder / 'training_result.pth')
                script['main']()
                constructor.assert_called_once_with(weights=weights)
            self.assertFalse(backbone.training)
            self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))
            self.assertTrue(torch.equal(batch_norm.running_mean, bn_mean))
            checkpoint = torch.load(folder / 'training_result.pth', weights_only=True)
            self.assertEqual(checkpoint['class_names'], ['blue', 'red'])
            self.assertEqual(checkpoint['model_state_dict']['1.weight'].shape, (4, width))

    def test_invalid_batch_size(self):
        node = self.node('DataLoaderNode', [torch.utils.data.TensorDataset(torch.ones(2, 1)), 0, False])
        node.update(); self.assertIsNone(self.output(node))

    def test_training_explicit_and_updates_parameters(self):
        x = torch.randn(12, 2); y = (x[:, 0] > 0).long()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=5)
        model = torch.nn.Linear(2, 2)
        before = model.weight.detach().clone()
        node = self.node('TrainModelNode', [model, loader, 'cross_entropy', 'Adam', .05, 4])
        node.update(inp=1)
        self.assertTrue(torch.equal(before, model.weight))
        history = node.train()
        self.assertEqual(len(history), 4)
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
        self.assertLess(history[-1], history[0])
        self.assertFalse(torch.equal(before, model.weight))
        self.assertIs(self.output(node), model)
        evaluation = self.node('TestModelNode', [model, loader])
        evaluation.update()
        self.assertIsInstance(self.output(evaluation), float)
        self.assertEqual(sum(len(b) for b in self.output(evaluation, 1)), 12)
        self.assertTrue(model.training)

    def test_losses_and_optimizers(self):
        for loss in training.LOSS_FUNCTIONS:
            x = torch.randn(4, 2)
            y = torch.tensor([0, 1, 0, 1]) if loss == 'cross_entropy' else torch.zeros(4, 2)
            model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid()) if loss == 'bce' else torch.nn.Linear(2, 2)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
            for optimizer in ('Adam', 'SGD', 'RMSprop', 'AdamW'):
                with self.subTest(loss=loss, optimizer=optimizer):
                    _, history = training.train_model(model, loader, loss, optimizer)
                    self.assertTrue(torch.isfinite(torch.tensor(history)).all())

    def test_training_rejects_bad_configuration(self):
        model = torch.nn.Linear(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(2, 2)))
        for kwargs in ({}, {'epochs': 0}, {'learning_rate': 0}, {'loss_key': 'bad'}, {'optimizer_name': 'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                training.train_model(model, loader, **kwargs)

    def test_empty_training_and_evaluation(self):
        model = torch.nn.Linear(2, 2)
        empty = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, 'empty'):
            training.train_model(model, empty)
        node = self.node('TestModelNode', [model, empty]); node.update()
        self.assertIsNone(self.output(node))

    def test_cancel_training_before_parameter_update(self):
        model = torch.nn.Linear(2, 2)
        before = model.weight.detach().clone()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)))
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            training.train_model(model, loader, cancelled=lambda: True)
        torch.testing.assert_close(model.weight, before)

    def test_regression_evaluation_and_missing_input(self):
        model = torch.nn.Linear(2, 1).double()
        x = torch.ones(3, 2); y = torch.zeros(3, 1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
        node = self.node('TestModelNode', [model, loader, 'mse']); node.update()
        with torch.no_grad(): expected = model(x.double()).square().mean().item()
        self.assertAlmostEqual(self.output(node), expected)
        node.inputs[0].default = Data(None); node.update()
        self.assertIsNone(self.output(node))
