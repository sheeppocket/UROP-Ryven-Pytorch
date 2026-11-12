"""Run separately: Qt must be initialized before importing the node package."""
import os
os.environ['RYVEN_MODE'] = 'gui'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyside2'
import json
from pathlib import Path
import sys
import time
import torch
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
train = nodes['TrainModelNode']
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

for filename, expected in (
    ('linear_demo.json', 'nn.Linear(2, 2)'),
    ('fashion_mnist_vae.json', 'class VAE(nn.Module):'),
    ('five_class_demo.json', "DATA_PATH = 'pytorch/demo_data/digits/train'"),
    ('weed_demo.json', "DATA_PATH = 'pytorch/demo_data/deepweeds/train'"),
):
    project = Path(__file__).resolve().parents[1] / filename
    demo, = session.load(json.loads(project.read_text(encoding='utf-8')))
    for node in demo.nodes:
        if isinstance(node, types['SampleDatasetNode']):
            node.gui.main_widget().load_button.click()
    trainer, = (node for node in demo.nodes if isinstance(node, (types['TrainModelNode'], types['TrainVAENode'])))
    app.clipboard().clear()
    trainer.gui.main_widget().copy_code_button.click()
    code = app.clipboard().text()
    assert expected in code, trainer.gui.main_widget().status_box.toPlainText()
    compile(code, filename, 'exec')
    session.delete_flow(demo)
assert not errors, errors
print('All four saved graphs opened and copied training code', flush=True)
print('GUI SMOKE TEST PASSED', flush=True)
