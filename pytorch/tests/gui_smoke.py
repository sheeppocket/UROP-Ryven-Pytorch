"""Run separately: Qt must be initialized before importing the node package."""
import os
os.environ['RYVEN_MODE'] = 'gui'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyside2'
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import torch
from PIL import Image
from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtCore import QTimer
from ryvencore import Data
from ryvencore_qt import SessionGUI
from ryven.main.packages.nodes_package import import_nodes_package

from ryven.main.config import Config
Config()
app = QApplication.instance() or QApplication([])
torch.set_num_threads(2)
errors = []
def exception_hook(kind, error, tb):
    errors.append(str(error))
    sys.__excepthook__(kind, error, tb)
sys.excepthook = exception_hook
classes, _ = import_nodes_package(directory=str(Path(__file__).resolve().parents[1] / 'nodes'))
types = {c.__name__: c for c in classes}
package = types['ModelsNode'].__module__.rsplit('.', 1)[0]
gui = importlib.import_module(package + '.gui')
parent = QWidget()
wrapper = SessionGUI(parent)
session = wrapper.core_session
session.register_node_types(classes)
flow = session.create_flow('GUI test')
nodes = {}
for name, cls in types.items():
    node = flow.create_node(cls)
    node.update_error.sub(lambda error: errors.append(str(error)))
    nodes[name] = node
app.processEvents()
print('Created all node widgets', flush=True)
# Confirm widget state round trips and JSON compatibility.
for node in nodes.values():
    if hasattr(node, 'gui'):
        for index in node.gui.init_input_widgets:
            widget = node.gui.input_widget(index)
            state = widget.get_state()
            json.dumps(state)
            widget.set_state(state)
print('Widget state round trips passed', flush=True)
dataset = nodes['TensorDatasetNode']
count = len(dataset.inputs)
dataset.gui.input_widget(0).click()
assert len(dataset.inputs) == count + 1
print('Add socket button passed', flush=True)
tensor = nodes['MakeTensorNode']
tensor.gui.input_widget(0).setText('(4, 2)')
tensor.gui.input_widget(2).combo.setCurrentIndex(3)
tensor.update()
assert tensor.outputs[0].val.payload.shape == (4, 2)
print('Tensor widget edits propagate correctly', flush=True)
model_node = nodes['ModelsNode']
model_node.gui.input_widget(1).setText('random')
item = wrapper.flow_views[flow].node_items[model_node]
item.main_widget.click()
assert isinstance(model_node.outputs[0].val.payload, torch.nn.Module)
print('Load Model button passed', flush=True)
image_split = nodes['ImageFolderSplitNode']
flow.connect_nodes(model_node.outputs[0], image_split.inputs[0])
picker = image_split.gui.input_widget(1)
load_images = image_split.gui.main_widget().load_button
preset = picker._popup.item(0)
picker._on_popup_item_clicked(preset)
load_images.click()
assert len(image_split.outputs[1].val.payload) == 125
assert len(image_split.outputs[2].val.payload) == 50
with tempfile.TemporaryDirectory(prefix='custom image data ') as folder:
    for label in ('a', 'b'):
        path = Path(folder) / label
        path.mkdir()
        for index in range(2):
            Image.new('RGB', (4, 4)).save(path / f'{index}.png')
    picker.setEditText(folder)
    state = picker.get_state()
    picker.set_state({'val': preset.text()})
    picker.set_state(state)
    assert picker.currentText() == folder
    load_images.click()
    assert image_split.outputs[0].val.payload == ('a', 'b')
    assert len(image_split.outputs[1].val.payload) == 2
    assert len(image_split.outputs[2].val.payload) == 2
    picker._on_popup_item_clicked(preset)
    load_images.click()
    assert len(image_split.outputs[1].val.payload) == 125
print('Editable image dropdown switched between bundled and custom folders', flush=True)
digits = nodes['DigitsDemoNode']
classifier = nodes['FrozenResNetClassifierNode']
flow.connect_nodes(digits.outputs[0], classifier.inputs[1])
flow.connect_nodes(model_node.outputs[0], classifier.inputs[0])
digits.gui.input_widget(0).setText('[9, 2]')
digits.gui.main_widget().load_button.click()
assert classifier.outputs[0].val.payload._ryven_categories == ('nine', 'two')
assert '512 features → 2 class scores' in classifier.gui.main_widget().text()
loader = nodes['DataLoaderNode']
export_train = nodes['TrainModelNode']
flow.connect_nodes(digits.outputs[1], loader.inputs[0])
flow.connect_nodes(loader.outputs[0], export_train.inputs[1])
flow.connect_nodes(classifier.outputs[0], export_train.inputs[0])
export_widget = export_train.gui.main_widget()
export_widget.copy_code_button.click()
copied_code = app.clipboard().text()
compile(copied_code, '<clipboard training export>', 'exec')
assert "'class_names': ('nine', 'two')" in copied_code
assert 'Training code copied' in export_widget.status_box.toPlainText()
digits.gui.input_widget(0).setText('[0, 1, 2, 3, 4]')
assert classifier.outputs[0].val.payload is None
export_widget.copy_code_button.click()
assert app.clipboard().text() == copied_code
assert 'Could not copy training code' in export_widget.status_box.toPlainText()
print('Digit choices, classifier setup, and clipboard export passed', flush=True)
dense_input = nodes['DenseInputNode']
dense_hidden = nodes['DenseLayerNode']
dense_output = flow.create_node(types['DenseLayerNode'])
dense_trainer = flow.create_node(types['TrainModelNode'])
csv_source = nodes['RemoteCsvDatasetNode']
csv_batches = flow.create_node(types['DataLoaderNode'])
flow.connect_nodes(dense_input.outputs[0], dense_hidden.inputs[0])
flow.connect_nodes(dense_hidden.outputs[0], dense_output.inputs[0])
flow.connect_nodes(dense_output.outputs[0], dense_trainer.inputs[0])
flow.connect_nodes(csv_source.outputs[0], csv_batches.inputs[0])
flow.connect_nodes(csv_batches.outputs[0], dense_trainer.inputs[1])
dense_input.gui.input_widget(0).setText('3')
dense_hidden.gui.input_widget(1).setText('6')
dense_hidden.gui.input_widget(2).setCurrentText('GELU')
dense_output.gui.input_widget(1).setText('2')
dense_output.gui.input_widget(2).setCurrentText('None')
csv_source.gui.input_widget(0).setText('/content/data.csv')
dense_trainer.gui.main_widget().copy_code_button.click()
dense_code = app.clipboard().text()
compile(dense_code, '<clipboard dense training export>', 'exec')
assert "DATA_PATH = '/content/data.csv'" in dense_code
assert 'nn.Linear(3, 6)' in dense_code
assert 'nn.GELU()' in dense_code
assert 'nn.Linear(6, 2)' in dense_code
print('Dense layer settings propagated to the clipboard export', flush=True)
evaluation = nodes['EvaluateModelNode']
flow.connect_nodes(model_node.outputs[0], evaluation.inputs[0])
evaluation_widget = wrapper.flow_views[flow].node_items[evaluation].main_widget
preview_checked = []
forward_calls = []
hook = model_node.outputs[0].val.payload.register_forward_hook(lambda *_: forward_calls.append(True))

def check_feature_dialog():
    dialog = app.activeModalWidget()
    try:
        assert isinstance(dialog, gui.FeatureMapsDialog)
        reference_picker = dialog.reference_picker
        picker = dialog.picker
        assert picker.itemText(0) == '1. conv1'
        assert reference_picker.count() == picker.count() == 20
        assert all(reference_picker.itemText(index) == picker.itemText(index) for index in range(picker.count()))
        assert reference_picker.currentIndex() == 0
        assert picker.currentIndex() == picker.count() - 1
        assert '112 × 112' in dialog.reference_layer.summary.text()
        assert '7 × 7' in dialog.selected_layer.summary.text()
        assert not dialog.reference_layer.tiles[0][1].pixmap().isNull()
        assert not dialog.selected_layer.tiles[0][1].pixmap().isNull()
        assert not dialog.next_button.isEnabled()
        calls_before_navigation = len(forward_calls)
        middle = picker.count() // 2
        picker.setCurrentIndex(middle)
        assert dialog.selected_layer.layer_name.text() == dialog.layers[middle][0]
        assert dialog.reference_layer.layer_name.text() == 'conv1'
        reference_picker.setCurrentIndex(reference_picker.count() - 1)
        assert '7 × 7' in dialog.reference_layer.summary.text()
        assert picker.currentIndex() == middle
        assert dialog.progress.text() == f'Layer {middle + 1} of 20'
        dialog.previous_button.click()
        assert picker.currentIndex() == middle - 1
        dialog.next_button.click()
        assert picker.currentIndex() == middle
        assert reference_picker.currentIndex() == reference_picker.count() - 1
        picker.setCurrentIndex(0)
        assert picker.currentIndex() == 0
        assert '112 × 112' in dialog.selected_layer.summary.text()
        assert not dialog.previous_button.isEnabled()
        assert len(forward_calls) == calls_before_navigation
        preview_checked.append(True)
    finally:
        if dialog is not None:
            dialog.reject()

with tempfile.TemporaryDirectory() as tmp:
    photo = Path(tmp) / 'input.png'
    Image.new('RGB', (31, 23), 'blue').save(photo)
    evaluation.gui.input_widget(1).setText(str(photo))
    evaluation_widget.run_button.click()
    assert not evaluation_widget.image_preview.isHidden()
    assert not evaluation_widget.image_preview.pixmap().isNull()
    QTimer.singleShot(0, check_feature_dialog)
    evaluation_widget.feature_button.click()
    assert preview_checked, evaluation_widget.report_box.toPlainText()
    chosen = Path(tmp) / 'chosen.png'
    Image.new('RGB', (23, 31), 'red').save(chosen)
    with patch.object(gui.QFileDialog, 'getOpenFileName', return_value=(str(chosen), '')):
        QTimer.singleShot(0, check_feature_dialog)
        evaluation_widget.file_button.click()
    assert len(preview_checked) == 2 and len(forward_calls) == 3
    assert evaluation.photo_preview.getpixel((0, 0)) == (255, 0, 0)
    assert evaluation.input(1).payload == str(photo)
    evaluation.gui.input_widget(1).setText('')
    assert evaluation_widget.image_preview.isHidden()
hook.remove()
print('Image selection, in-node preview, and convolution layer comparison passed', flush=True)
train = nodes['TrainModelNode']
flow.disconnect_nodes(classifier.outputs[0], train.inputs[0])
flow.disconnect_nodes(loader.outputs[0], train.inputs[1])
model = torch.nn.Linear(2, 2)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(20, 2), torch.arange(20) % 2), batch_size=4)
for i, val in enumerate((model, loader, 'cross_entropy', 'Adam', .01, 3)):
    train.inputs[i].default = Data(val)
widget = wrapper.flow_views[flow].node_items[train].main_widget
widget.on_train()
ticks = []
timer = QTimer(); timer.timeout.connect(lambda: ticks.append(1)); timer.start(1)
deadline = time.monotonic() + 30
while not widget.train_button.isEnabled() and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(.001)
widget.worker.wait(30000)
app.processEvents(); timer.stop()
assert train.outputs[0].val.payload is model, widget.status_box.toPlainText()
assert len(train.outputs[1].val.payload) == 3
assert ticks, 'Qt event loop did not run while training'
# Exercise a worker failure and confirm the Train button becomes usable again.
train.inputs[1].default = Data(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(2, 2))))
widget.on_train()
deadline = time.monotonic() + 30
while not widget.train_button.isEnabled() and time.monotonic() < deadline:
    app.processEvents(); time.sleep(.001)
widget.worker.wait(30000); app.processEvents()
assert 'requires' in widget.status_box.toPlainText(), widget.status_box.toPlainText()
assert train.outputs[0].val.payload is None
assert widget.train_button.isEnabled()
assert not errors, errors
print('Background training, main-thread result publication, and live event loop passed', flush=True)

for filename in ('five_class_demo.json', 'weed_demo.json'):
    project = Path(__file__).resolve().parents[1] / filename
    demo, = session.load(json.loads(project.read_text(encoding='utf-8')))
    backbone, = (node for node in demo.nodes if isinstance(node, types['ModelsNode']))
    folders, = (node for node in demo.nodes if isinstance(node, types['ImageFolderPathsNode']))
    backbone.gui.main_widget().click()
    folders.gui.main_widget().load_button.click()
    reports = {node.gui.main_widget().text() for node in demo.nodes
               if isinstance(node, types['ImagesToTensorNode'])}
    assert any('(125, 3, 224, 224)' in report for report in reports), reports
    assert any('(50, 3, 224, 224)' in report for report in reports), reports
    session.delete_flow(demo)
assert not errors, errors
print('Saved CNN graphs loaded and displayed both tensor shapes', flush=True)
print('GUI SMOKE TEST PASSED', flush=True)


