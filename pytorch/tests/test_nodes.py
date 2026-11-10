"""Integration tests using real Ryven nodes, ports and flow propagation."""
import os
os.environ['RYVEN_MODE'] = 'no-gui'
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image
from ryvencore import Data, Node, NodeOutputType, Session
from ryven.main.packages.nodes_package import import_nodes_package

ROOT = Path(__file__).resolve().parents[1]
CLASSES, _ = import_nodes_package(directory=str(ROOT / 'nodes'))
TYPES = {cls.__name__: cls for cls in CLASSES}
PACKAGE = TYPES['ModelsNode'].__module__.rsplit('.', 1)[0]
training = importlib.import_module(PACKAGE + '.train_model')

class Source(Node):
    init_outputs = [NodeOutputType()]

class NodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.session = Session()
        self.session.register_node_types(CLASSES + [Source])
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

    def send(self, node, index, item):
        source = self.flow.create_node(Source)
        self.flow.connect_nodes(source.outputs[0], node.inputs[index])
        source.set_output_val(0, Data(item))
        self.assertEqual(self.errors, [])
        return source

    def test_all_exported_callbacks_and_serialization(self):
        for name in TYPES:
            node = self.node(name)
            node.update(inp=0)
        self.assertEqual(self.errors, [])
        saved = self.session.serialize()
        json.dumps(saved)
        other = Session()
        other.register_node_types(CLASSES + [Source])
        other.load(saved)
        self.assertEqual(len(other.flows[0].nodes), len(TYPES))

    def test_saved_learning_presets_train_and_evaluate(self):
        for filename, feature_shape, epochs in (
            ('linear_demo.json', (2,), 8),
            ('neural_network_demo.json', (2,), 14),
            ('rnn_demo.json', (6, 1), 16),
        ):
            with self.subTest(preset=filename):
                session = Session()
                session.register_node_types(CLASSES)
                flow, = session.load(json.loads((ROOT / filename).read_text(encoding='utf-8')))
                nodes = {type(node).__name__: node for node in flow.nodes}
                data = nodes['SampleDatasetNode']
                model = nodes['SampleModelNode']
                data.update()
                model.update()
                train_data, test_data = self.output(data, 0), self.output(data, 1)
                self.assertEqual(len(train_data), 128)
                self.assertEqual(len(test_data), 64)
                self.assertEqual(tuple(train_data[0][0].shape), feature_shape)
                self.assertFalse(torch.equal(train_data[0][0], test_data[0][0]))
                initial_weights = {
                    key: weight.clone() for key, weight in self.output(model).state_dict().items()
                }
                history = nodes['TrainModelNode'].train()
                self.assertEqual(len(history), epochs)
                self.assertLess(history[-1], history[0])
                self.assertGreaterEqual(nodes['TestModelNode'].accuracy, .95)
                model.update()
                for key, weight in self.output(model).state_dict().items():
                    self.assertTrue(torch.equal(weight, initial_weights[key]))

    def test_tensor_dtypes_and_fills(self):
        for dtype in ('float32', 'float64', 'float16', 'int32', 'int64', 'bool'):
            for fill in ('zeros', 'ones', '3'):
                with self.subTest(dtype=dtype, fill=fill):
                    node = self.node('MakeTensorNode', [(2, 3), dtype, fill])
                    node.update()
                    result = self.output(node)
                    self.assertEqual(result.shape, (2, 3))
                    self.assertEqual(result.dtype, getattr(torch, dtype))
        for fill in ('rand', 'randn'):
            node = self.node('MakeTensorNode', [(2, 3), 'float32', fill])
            node.update(); self.assertTrue(torch.isfinite(self.output(node)).all())

    def test_bad_tensor_clears_previous_value(self):
        node = self.node('MakeTensorNode', [(2,), 'float32', 'ones'])
        node.update(); self.assertIsNotNone(self.output(node))
        for shape, dtype, fill in [((-1,), 'float32', 'ones'), ((2,), 'int64', 'randn'), ((2,), 'bad', 'ones')]:
            for i, val in enumerate((shape, dtype, fill)): node.inputs[i].default = Data(val)
            node.update(); self.assertIsNone(self.output(node))

    def test_connected_data_pipeline_and_disconnect(self):
        dataset = self.node('TensorDatasetNode')
        dataset.create_input(label='tensor 2')
        x = torch.randn(7, 2); y = torch.arange(7)
        self.send(dataset, 1, x); labels = self.send(dataset, 2, y)
        self.assertEqual(len(self.output(dataset)), 7)
        loader = self.node('DataLoaderNode', [None, 3, 'False'])
        self.flow.connect_nodes(dataset.outputs[0], loader.inputs[0])
        batches = list(self.output(loader))
        self.assertEqual([len(b[0]) for b in batches], [3, 3, 1])
        self.assertTrue(torch.equal(torch.cat([b[1] for b in batches]), y))
        labels.set_output_val(0, Data(torch.arange(2)))
        self.assertIsNone(self.output(dataset))
        self.assertIsNone(self.output(loader))
        self.assertIsNone(loader.dataloader)

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

    def test_images_and_inference_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'; Image.new('RGB', (32, 24), 'red').save(path)
            node = self.node('ImagesToTensorNode', [[str(path)], 16, 'False'])
            node.update(); result = self.output(node)
            self.assertEqual(result.shape, (1, 3, 16, 16))
            self.assertEqual(result[0, 0, 0, 0].item(), 1)
            node.inputs[0].default = Data('missing.png'); node.update()
            self.assertIsNone(self.output(node))
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
            report = self.node('EvaluateModelNode', [model, str(path)])
            report.request_run()
            self.assertIn('class', report.report)
            self.assertNotIn('failed', report.report)
            self.assertTrue(model.training)
            report.inputs[1].default = Data('missing.png'); report.request_run()
            self.assertIn('failed', report.report)

    def test_feature_maps_capture_raw_outputs_and_remove_hooks(self):
        first = torch.nn.Conv2d(3, 10, 1)
        last = torch.nn.Conv2d(10, 2, 1, stride=2)
        with torch.no_grad():
            first.weight.zero_()
            first.bias.fill_(-2)
        model = torch.nn.Sequential(first, torch.nn.ReLU(inplace=True), last,
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        model._ryven_transform = lambda image: torch.ones(3, 9, 11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'photo.png'
            Image.new('RGB', (13, 7), 'red').save(path)
            node = self.node('EvaluateModelNode', [model, path])
            node.request_run()
            predictions = node.report
            self.assertIsNone(node.feature_preview)
            node.request_run(show_feature_maps=True)
            self.assertEqual(node.report, predictions)
            photo, layers = node.feature_preview
            self.assertEqual(photo.size, (13, 7))
            self.assertEqual([(name, shape) for name, shape, _ in layers],
                             [('0', (10, 9, 11)), ('2', (2, 5, 6))])
            torch.testing.assert_close(layers[0][2], torch.full((8, 9, 11), -2.0))
            self.assertTrue(model.training)
            self.assertFalse(first._forward_hooks)
            self.assertFalse(last._forward_hooks)

            model[2] = torch.nn.Conv2d(3, 2, 1)  # Fails after the first hook captures a layer.
            node.request_run(show_feature_maps=True)
            self.assertIn('Inference failed', node.report)
            self.assertIsNone(node.feature_preview)
            self.assertTrue(model.training)
            self.assertFalse(first._forward_hooks)
            self.assertFalse(model[2]._forward_hooks)

    def test_single_image_label_check_and_dataset_accuracy(self):
        head = torch.nn.Linear(3, 2)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([2., 0.]))
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), head)
        model._ryven_categories = ('red', 'blue')
        photo = Image.new('RGB', (13, 7), 'red')
        singleton = self.node('EvaluateModelNode', [model, photo, 'red'])
        singleton.request_run()
        self.assertIn('Expected red · Correct prediction', singleton.report)

        label = self.send(singleton, 2, 'blue')
        singleton.request_run()
        self.assertIn('Expected blue · Incorrect prediction', singleton.report)
        singleton.request_run(image=photo)
        self.assertNotIn('Expected', singleton.report)
        label.set_output_val(0, Data(7))
        singleton.request_run()
        self.assertIn('index from 0 to 1', singleton.report)
        self.assertIn('1. red', singleton.report)

        dataset = torch.utils.data.TensorDataset(torch.zeros(3, 3, 4, 4), torch.tensor([0, 1, 0]))
        test = self.node('TestModelNode', [model, torch.utils.data.DataLoader(dataset, batch_size=2)])
        test.update()
        self.assertEqual(test.accuracy, 2 / 3)
        self.assertEqual([len(batch) for batch in self.output(test, 1)], [2, 1])

    def test_image_folder_split_keeps_class_mapping_and_test_images_separate(self):
        from torchvision.transforms import ToTensor

        model = torch.nn.Linear(3, 2)
        model._ryven_transform = ToTensor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'photos'
            explicit_test = Path(tmp) / 'held out'
            for name in ('sedge', 'grass'):
                (root / name).mkdir(parents=True)
                for index in range(5):
                    Image.new('RGB', (9, 7), (index, 2, 3)).save(root / name / f'{index}.png')
                (explicit_test / name).mkdir(parents=True)
                Image.new('RGB', (9, 7), 'red').save(explicit_test / name / 'sample.png')

            node = self.node('ImageFolderSplitNode', [model, root])
            node.update()
            self.assertEqual(self.output(node), ('grass', 'sedge'))
            train, test = self.output(node, 1), self.output(node, 2)
            self.assertEqual((len(train), len(test)), (8, 2))
            self.assertTrue(set(train.indices).isdisjoint(test.indices))
            self.assertEqual([test[i][1] for i in range(len(test))], [0, 1])
            self.assertIs(train.dataset.transform, model._ryven_transform)
            self.assertEqual(Path(self.output(node, 3)), Path(test.dataset.samples[test.indices[0]][0]))
            node.update()
            self.assertEqual(self.output(node, 2).indices, test.indices)

            node.inputs[2].default = Data(explicit_test)
            node.update(inp=2)
            self.assertIsNone(self.output(node, 3))
            node.update()
            train, test = self.output(node, 1), self.output(node, 2)
            self.assertEqual((len(train), len(test)), (10, 2))
            self.assertEqual(train.class_to_idx, test.class_to_idx)
            self.assertEqual(Path(self.output(node, 3)).parent, explicit_test / 'grass')
            self.assertEqual(self.output(node, 4), 'grass')

            different_classes = Path(tmp) / 'different classes'
            (different_classes / 'other').mkdir(parents=True)
            Image.new('RGB', (9, 7), 'red').save(different_classes / 'other' / 'sample.png')
            node.inputs[2].default = Data(different_classes)
            node.update()
            self.assertIn('same class names', node.report)
            self.assertTrue(all(self.output(node, index) is None for index in range(5)))
            node.inputs[2].default = Data(root)
            node.update()
            self.assertIn('separate directories', node.report)

    def test_cnn_demos_wire_image_and_label_tensors_and_preserve_training_export(self):
        for filename in ('five_class_demo.json', 'weed_demo.json'):
            with self.subTest(preset=filename):
                session = Session()
                session.register_node_types(CLASSES)
                flow, = session.load(json.loads((ROOT / filename).read_text(encoding='utf-8')))
                by_type = {type(node).__name__: node for node in flow.nodes}
                folders = by_type['ImageFolderPathsNode']
                trainer = by_type['TrainModelNode']
                folders.update()
                self.assertIsNone(trainer.training_inputs()[1])
                by_type['ModelsNode'].update()

                training_batches = trainer.training_inputs()[1]
                test_batches = by_type['TestModelNode'].input(1).payload
                for batches, size, label_port in ((training_batches, 125, 2), (test_batches, 50, 4)):
                    self.assertIsInstance(batches.dataset, torch.utils.data.TensorDataset)
                    images, labels = batches.dataset.tensors
                    self.assertEqual(images.shape, (size, 3, 224, 224))
                    self.assertEqual(labels.dtype, torch.long)
                    torch.testing.assert_close(labels, self.output(folders, label_port))

                script = {'__name__': 'training_export'}
                exec(compile(trainer.export_training_code(), '<training export>', 'exec'), script)
                script['DATA_ROOT'] = self.output(folders, 1).root
                exported = script['make_training_dataset']()
                self.assertEqual(len(exported), len(training_batches.dataset))
                for index in (0, 25, 124):
                    image, label = training_batches.dataset[index]
                    exported_image, exported_label = exported[index]
                    torch.testing.assert_close(image, exported_image)
                    self.assertEqual(label.item(), exported_label)

                folders.update(inp=0)
                self.assertIsNone(trainer.training_inputs()[1])
                self.assertIsNone(by_type['TestModelNode'].input(1).payload)

    def test_digit_selection_maps_nonconsecutive_digits_to_contiguous_labels(self):
        model = torch.nn.Linear(3, 2)
        model._ryven_transform = lambda image: torch.ones(3, 11, 13)
        node = self.node('DigitsDemoNode', [(9, 2, 7), model])
        node.update()
        self.assertEqual(self.output(node), ('nine', 'two', 'seven'))
        train, test = self.output(node, 1), self.output(node, 2)
        self.assertEqual(train.labels.tolist(), [0] * 25 + [1] * 25 + [2] * 25)
        self.assertEqual(test.labels.tolist(), [0] * 10 + [1] * 10 + [2] * 10)
        self.assertEqual(train[0][0].shape, (3, 11, 13))
        self.assertEqual(test[0][0].shape, (3, 11, 13))
        self.assertEqual(self.output(node, 3).tobytes(), test.image(0).tobytes())
        self.assertEqual(self.output(node, 4), 'nine')

        node.inputs[0].default = Data([9, 9])
        node.update(inp=0)
        self.assertTrue(all(self.output(node, index) is None for index in range(5)))
        node.update()
        self.assertIn('distinct digits', node.report)

    def test_digit_node_restores_default_selection_in_older_projects(self):
        data = self.node('DigitsDemoNode').data()
        data['inputs'] = []
        restored = self.flow.create_node(TYPES['DigitsDemoNode'], data)
        self.assertEqual(restored.inputs[0].default.payload, (0, 1, 2, 3, 4))
        self.assertEqual(restored.inputs[1].label_str, 'model')

    def test_random_resnet_forward_and_custom_weight_reload(self):
        node = self.node('ModelsNode', ['resnet18', 'random'])
        node.update(); model = self.output(node)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad(): expected = model(x)
        self.assertEqual(expected.shape, (1, 1000))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pth'; torch.save(model.state_dict(), path)
            restored = node.get_model('resnet18', str(path)); restored.eval()
            with torch.no_grad(): torch.testing.assert_close(restored(x), expected)
        node.inputs[0].default = Data('bad'); node.update()
        self.assertIsNone(self.output(node)); self.assertIsNone(node.model)

    def test_frozen_resnet_trains_only_the_custom_head(self):
        source = self.node('ModelsNode', ['resnet18', 'random'])
        source.update()
        original = self.output(source)
        original._ryven_transform = torch.nn.Identity()
        names = ('zero', 'one', 'two', 'three', 'four')
        frozen = self.node('FrozenResNetClassifierNode', [None, names])
        self.flow.connect_nodes(source.outputs[0], frozen.inputs[0])
        model = self.output(frozen)
        encoder_state = {key: tensor.clone() for key, tensor in model.encoder.state_dict().items()}
        self.assertEqual(original.fc.out_features, 1000)
        self.assertTrue(original.conv1.weight.requires_grad)
        self.assertTrue(original._ryven_transform.training)
        self.assertIsNot(model._ryven_transform, original._ryven_transform)
        self.assertEqual(model._ryven_categories, names)
        self.assertEqual(model.classifier.out_features, len(names))
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                         ['classifier.weight', 'classifier.bias'])

        head_before = model.classifier.weight.detach().clone()
        inputs = torch.randn(5, 3, 32, 32)
        labels = torch.arange(5)
        dataset = self.node('TensorDatasetNode', [None, (inputs, labels)])
        loader = self.node('DataLoaderNode', [None, 2])
        train = self.node('TrainModelNode')
        test = self.node('TestModelNode')
        self.flow.connect_nodes(dataset.outputs[0], loader.inputs[0])
        self.flow.connect_nodes(loader.outputs[0], train.inputs[1])
        self.flow.connect_nodes(loader.outputs[0], test.inputs[1])
        self.flow.connect_nodes(frozen.outputs[0], train.inputs[0])
        self.flow.connect_nodes(train.outputs[0], test.inputs[0])
        dataset.update()
        train.train()
        self.assertIs(self.output(train), model)
        self.assertIsNotNone(test.accuracy)
        self.assertTrue(model.classifier.training)
        self.assertTrue(original._ryven_transform.training)
        self.assertTrue(all(not layer.training for layer in model.encoder.modules()))
        self.assertFalse(torch.equal(model.classifier.weight, head_before))
        for key, tensor in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(tensor, encoder_state[key]), key)

        frozen.inputs[1].default = Data(('one', 'one'))
        frozen.update(inp=1)
        self.assertIsNone(self.output(frozen))
        self.assertIsNone(self.output(train))
        self.assertIsNone(test.accuracy)

    def test_existing_image_tensor_node_uses_connected_preprocessing(self):
        photo = Image.new('RGB', (9, 7), 'red')
        model = torch.nn.Linear(3, 2)
        model._ryven_transform = lambda image: torch.full((3, 11, 13), image.getpixel((0, 0))[0] / 255)
        node = self.node('ImagesToTensorNode', [photo, 16, False])
        node.update()
        self.assertEqual(self.output(node).shape, (1, 3, 16, 16))
        source = self.send(node, 3, model)
        torch.testing.assert_close(self.output(node), torch.ones(1, 3, 11, 13))
        source.set_output_val(0, Data(None))
        self.assertIsNone(self.output(node))

    def test_image_to_trained_model_connected_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, color in enumerate(('red', 'blue', 'red', 'blue')):
                path = Path(tmp) / f'{index}.png'
                Image.new('RGB', (16, 16), color).save(path)
                paths.append(str(path))
            images = self.node('ImagesToTensorNode', [paths, 16, False])
            dataset = self.node('TensorDatasetNode')
            dataset.create_input(label='tensor 2')
            self.send(dataset, 2, torch.tensor([0, 1, 0, 1]))
            loader = self.node('DataLoaderNode', [None, 2, False])
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
            train = self.node('TrainModelNode', [model, None, 'cross_entropy', 'Adam', .05, 3])
            evaluation = self.node('TestModelNode')
            self.flow.connect_nodes(images.outputs[0], dataset.inputs[1])
            self.flow.connect_nodes(dataset.outputs[0], loader.inputs[0])
            self.flow.connect_nodes(loader.outputs[0], train.inputs[1])
            self.flow.connect_nodes(loader.outputs[0], evaluation.inputs[1])
            self.flow.connect_nodes(train.outputs[0], evaluation.inputs[0])
            images.update()
            train.train()
            self.assertIsInstance(self.output(evaluation), float)
            self.assertEqual(sum(len(batch) for batch in self.output(evaluation, 1)), 4)
            self.assertEqual(self.errors, [])

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

    def test_invalid_weights_and_model_selection(self):
        node = self.node('ModelsNode')
        with self.assertRaises(ValueError): node.get_model('missing', 'random')
        with self.assertRaises((KeyError, ValueError)): node.get_model('resnet18', 'INVALID')
        node.inputs[1].default = Data('random'); node.update()
        self.assertIsNotNone(self.output(node))
        node.update(inp=1)
        self.assertIsNone(self.output(node))

    def test_preset_weights(self):
        node = self.node('PresetWeightsNode', ['resnet18']); node.update()
        self.assertEqual(self.output(node), 'DEFAULT')

if __name__ == '__main__':
    unittest.main(verbosity=2)
