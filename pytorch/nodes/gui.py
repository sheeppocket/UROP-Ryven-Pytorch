import ast
from typing import Tuple

from qtpy.QtGui import QFont
from qtpy.QtCore import Qt, Signal, QThread
from qtpy.QtWidgets import (QApplication, QPushButton, QComboBox, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QTextEdit, QWidget, QVBoxLayout, QLineEdit, QLabel)

from ryven.gui_env import *

from . import dense_layers, learning_samples, vae
from . import test_model as test_model_nodes
from . import torch_utils_data as torch_utils_data_nodes
from . import train_model as train_model_nodes


class PyTorchGuiBase(NodeGUI):
    color = '#5e548e'

def combo_box(items: list, init_index: int = 0, descr: str = ''):
    """Custom combo box widget with global popup, styled like QComboBox."""
    class StdInpWidget_ComboBox(NodeInputWidget, QComboBox):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QComboBox.__init__(self)
            self._prevent_update = False
            self.addItems(items)
            self.setCurrentIndex(init_index)
            self.currentTextChanged.connect(self.text_changed)
            self.setMinimumWidth(200)
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(9)
            self.setFont(font)
            self.setStyleSheet("""
                QComboBox {
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 2px 5px;
                    background: #2c2c2c;
                    color: #ddd;
                }
                QComboBox:hover {
                    border: 1px solid #5e548e;
                }
                QComboBox::drop-down {
                    border: none;
                    background: #2c2c2c;
                    width: 18px;
                }
            """)
            if descr:
                self.setToolTip(descr)

            # Custom global popup
            app = QApplication.instance()
            main_win = app.activeWindow() if app else None
            self._popup = QListWidget(main_win)
            self._popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self._popup.setStyleSheet("""
                QListWidget {
                    background: #2c2c2c;
                    color: #ddd;
                    border: 1px solid #555;
                    border-radius: 3px;
                    font-size: 9pt;
                }
                QListWidget::item {
                    padding: 3px 8px;
                    height: 22px;
                }
                QListWidget::item:selected {
                    background: #5e548e;
                    color: #fff;
                }
                QListWidget::item:hover {
                    background: #3d3d5c;
                }
            """)
            self._popup.setFont(font)
            self._popup.setMinimumWidth(200)
            self._popup.setMaximumWidth(200)
            self._popup.clear()
            for item in items:
                self._popup.addItem(QListWidgetItem(item))
            self._popup.itemClicked.connect(self._on_popup_item_clicked)

        def showPopup(self):
            # Align popup to QComboBox left edge, place directly below
            rect = self.rect()
            from qtpy.QtCore import QPoint
            global_pos = self.mapToGlobal(QPoint(rect.left(), rect.bottom()))
            self._popup.setMinimumWidth(200)
            self._popup.setMaximumWidth(200)
            # Force popup to fit all items, avoid empty last option
            self._popup.setSizeAdjustPolicy(QListWidget.AdjustToContents)
            self._popup.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            item_height = self._popup.sizeHintForRow(0)
            option_count = len(items)
            max_visible = min(option_count, 6)
            height = item_height * max_visible + 4
            self._popup.resize(width, height)
            self._popup.move(global_pos.x(), global_pos.y())
            self._popup.show()
            self._popup.raise_()
            self._popup.activateWindow()

        def hidePopup(self):
            self._popup.hide()

        def _on_popup_item_clicked(self, item):
            self.setCurrentIndex(self.findText(item.text()))
            self._popup.hide()

        def text_changed(self, text):
            if not self._prevent_update:
                self.update_node_input(Data(text))

        @property
        def val(self):
            return Data(self.currentText())

        def get_state(self):
            return {'val': self.val.payload}

        def set_state(self, data):
            if 'val' in data:
                text = data['val']
                text = text.payload if isinstance(text, Data) else text
                self._prevent_update = True
                index = self.findText(str(text))
                if index >= 0:
                    self.setCurrentIndex(index)
                self._prevent_update = False

        def val_update_event(self, val):
            self.set_state({'val': val})

    StdInpWidget_ComboBox.__doc__ = descr
    return StdInpWidget_ComboBox


def evaled_line_edit(init_text: str = '', descr: str = ''):
    """Creates a styled line edit input widget class for evaluated expressions.
    
    Args:
        init_text: initial text value
        descr: tooltip description
    """
    
    class StdInpWidget_EvaledLineEdit(NodeInputWidget, QLineEdit):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QLineEdit.__init__(self)
            
            self._prevent_update = False
            
            self.setText(init_text)
            
            # Set styles to match combo boxes
            self.setMinimumWidth(120)
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(9)
            self.setFont(font)
            
            self.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 2px 5px;
                    background: #2c2c2c;
                    color: #ddd;
                }
                QLineEdit:hover {
                    border: 1px solid #5e548e;
                }
                QLineEdit:focus {
                    border: 1px solid #5e548e;
                    background: #333;
                }
            """)
            
            if descr:
                self.setToolTip(descr)
            
            self.textChanged.connect(self.text_changed)
        
        def text_changed(self, text):
            if not self._prevent_update:
                try:
                    val = ast.literal_eval(text) if text.strip() else None
                    self.update_node_input(Data(val))
                except Exception:
                    self.update_node_input(Data(text))
        
        @property
        def val(self):
            try:
                return Data(ast.literal_eval(self.text()) if self.text().strip() else None)
            except Exception:
                return Data(self.text())
        
        def get_state(self):
            return {'val': self.text()}
        
        def set_state(self, data):
            if 'val' in data:
                self._prevent_update = True
                self.setText(str(data['val']))
                self._prevent_update = False
        
        def val_update_event(self, val):
            self._prevent_update = True
            self.setText(str(val.payload) if val is not None and val.payload is not None else '')
            self._prevent_update = False
    
    StdInpWidget_EvaledLineEdit.__doc__ = descr
    return StdInpWidget_EvaledLineEdit


def float_spinbox(init: float = 1.0, range: Tuple[float, float] = (0.0, 99.0), 
                  decimals: int = 2, step: float = 0.1, descr: str = ''):
    """Creates a Float Spin Box input widget class."""
    
    class StdInpWidget_FloatSpinBox(NodeInputWidget, QDoubleSpinBox):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QDoubleSpinBox.__init__(self)
            
            self._prevent_update = False
            
            # Setup from widget_data
            self.setRange(*range)
            self.setDecimals(decimals)
            self.setSingleStep(step)
            self.setValue(init)
            
            # Set styles
            self.setMinimumWidth(100)
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(9)
            self.setFont(font)
            
            # Tooltip
            if descr:
                self.setToolTip(descr)
            
            self.valueChanged.connect(self.value_changed)
            
        def value_changed(self, value):
            if not self._prevent_update:
                self.update_node_input(Data(value))
                
        @property
        def val(self):
            return Data(self.value())
            
        def get_state(self):
            return {'val': self.val.payload}
            
        def set_state(self, data):
            if 'val' in data:
                self._prevent_update = True
                v = data['val']
                self.setValue(v.payload if isinstance(v, Data) else v)
                self._prevent_update = False
                
        def val_update_event(self, val):
            if isinstance(val.payload, (int, float)):
                self._prevent_update = True
                self.setValue(float(val.payload))
                self._prevent_update = False
    
    StdInpWidget_FloatSpinBox.__doc__ = descr
    return StdInpWidget_FloatSpinBox


class NodeReportMainWidget(NodeMainWidget, QLabel):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QLabel.__init__(self, self.node.instructions)
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setFixedSize(200, 80)
        self.node.report_changed.sub(self.show_report)

    def show_report(self, report):
        self.setText(report)
        self.setToolTip(report)


@node_gui(torch_utils_data_nodes.DataLoaderNode)
class DataLoaderNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'batch_size': evaled_line_edit('32', 'Batch size for data loading'),
        'shuffle': combo_box(
            items=[
                'False',
                'True'
            ],
            init_index=0,
            descr='Whether to shuffle the data at every epoch',
        ),
    }
    init_input_widgets = {
        1: {'name': 'batch_size', 'pos': 'besides'},
        2: {'name': 'shuffle', 'pos': 'besides'}
    }


@node_gui(torch_utils_data_nodes.RemoteCsvDatasetNode)
class RemoteCsvDatasetNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'csv_path': evaled_line_edit('train.csv', 'CSV path on the machine running the generated script'),
    }
    init_input_widgets = {0: {'name': 'csv_path', 'pos': 'besides'}}


@node_gui(dense_layers.DenseInputNode)
class DenseInputNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'features': evaled_line_edit('2', 'Number of input features per example'),
    }
    init_input_widgets = {0: {'name': 'features', 'pos': 'besides'}}


@node_gui(torch_utils_data_nodes.RemoteImageFolderNode)
class RemoteImageFolderNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'image_folder': evaled_line_edit('train', 'Class folders on the machine running the generated script'),
    }
    init_input_widgets = {0: {'name': 'image_folder', 'pos': 'besides'}}


@node_gui(torch_utils_data_nodes.FashionMNISTNode)
class FashionMNISTNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'data_root': evaled_line_edit('fashion-mnist-data', 'Download folder on the training machine'),
    }
    init_input_widgets = {0: {'name': 'data_root', 'pos': 'besides'}}


@node_gui(dense_layers.ImageEncoderNode)
class ImageEncoderNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'architecture': combo_box(list(dense_layers.RESNET_FEATURES), descr='ResNet to load in the generated script'),
        'weights': combo_box(list(dense_layers.ENCODER_WEIGHTS), descr='Pretrained or random initialization'),
        'freeze': combo_box(['True', 'False'], descr='Keep the encoder weights fixed during training'),
    }
    init_input_widgets = {
        0: {'name': 'architecture', 'pos': 'besides'},
        1: {'name': 'weights', 'pos': 'besides'},
        2: {'name': 'freeze', 'pos': 'besides'},
    }


@node_gui(dense_layers.DenseLayerNode)
class DenseLayerNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'neurons': evaled_line_edit('8', 'Number of neurons in this layer'),
        'activation': combo_box(list(dense_layers.ACTIVATIONS), init_index=1, descr='Activation after this layer'),
    }
    init_input_widgets = {
        1: {'name': 'neurons', 'pos': 'besides'},
        2: {'name': 'activation', 'pos': 'besides'},
    }


@node_gui(vae.VariationalLatentNode)
class VariationalLatentNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'latent_dimensions': evaled_line_edit('8', 'Number of values in the latent vector'),
    }
    init_input_widgets = {1: {'name': 'latent_dimensions', 'pos': 'besides'}}


class SampleDataMainWidget(NodeMainWidget, QWidget):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QWidget.__init__(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.load_button = QPushButton('Load Examples', self)
        layout.addWidget(self.load_button)
        self.report = QLabel(self.node.report, self)
        self.report.setTextFormat(Qt.PlainText)
        self.report.setWordWrap(True)
        self.report.setFixedWidth(204)
        layout.addWidget(self.report)
        self.load_button.clicked.connect(self.on_load)
        self.node_gui.updating.connect(self.reset_report)

    def reset_report(self):
        self.show_report(self.node.instructions)

    def on_load(self):
        self.node.update()
        self.show_report(self.node.report)

    def show_report(self, report):
        self.report.setText(report)
        self.report.setToolTip(report)
        self.report.setMinimumHeight(self.report.heightForWidth(self.report.width()))
        self.update_node_shape()


@node_gui(learning_samples.SampleDatasetNode)
class SampleDatasetNodeGui(PyTorchGuiBase):
    main_widget_class = SampleDataMainWidget
    input_widget_classes = {
        'sample': combo_box(list(learning_samples.SampleDatasetNode.options), descr='Sample dataset'),
    }
    init_input_widgets = {0: {'name': 'sample', 'pos': 'besides'}}


@node_gui(test_model_nodes.TestModelNode)
class TestModelNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'


_training_workers = set()


def _release_training_workers():
    for worker in list(_training_workers):
        if worker.isFinished():
            QApplication.instance().aboutToQuit.disconnect(worker.stop)
            worker.training_inputs = ()
            _training_workers.discard(worker)


class TrainingWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, inputs):
        super().__init__()
        self.training_inputs = inputs
        QApplication.instance().aboutToQuit.connect(self.stop)

    def stop(self):
        self.requestInterruption()
        self.wait()

    def run(self):
        from .train_model import train_configured_model
        try:
            model, history = train_configured_model(*self.training_inputs,
                cancelled=self.isInterruptionRequested,
                progress=lambda epoch, epochs, loss: self.progress.emit(
                    f'Epoch {epoch}/{epochs} loss={loss:.4f}'))
            self.succeeded.emit(model, history)
        except Exception as error:
            self.failed.emit(str(error))


class ExportTrainingMainWidget(NodeMainWidget, QWidget):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QWidget.__init__(self)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self.copy_code_button = QPushButton('Copy Training Code', self)
        self.copy_code_button.setFixedHeight(26)
        self.copy_code_button.setFont(QFont('Arial', 9))
        self.copy_code_button.setStyleSheet(
            "QPushButton{background:#3d3d5c;color:#ddd;border:1px solid #555;border-radius:3px;}"
            "QPushButton:hover{border-color:#5e548e;}"
            "QPushButton:pressed{background:#5e548e;}"
        )
        self.status_box = QTextEdit(self)
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(80)
        self.status_box.setStyleSheet(
            "QTextEdit{border:1px solid #555;border-radius:3px;padding:4px;background:#1e1e1e;color:#ddd;}"
        )
        self._layout.addWidget(self.copy_code_button)
        self._layout.addWidget(self.status_box)
        self.copy_code_button.clicked.connect(self.on_copy_training_code)

    def on_copy_training_code(self):
        try:
            code = self.node.export_training_code()
        except Exception as error:
            self.status_box.append(f'Could not copy training code. {error}')
            return
        QApplication.clipboard().setText(code)
        self.status_box.append('Training code copied to the clipboard.')

    def get_state(self):
        return {'status': self.status_box.toPlainText()}

    def set_state(self, data):
        if 'status' in data:
            self.status_box.setPlainText(str(data['status']))


class TrainMainWidget(ExportTrainingMainWidget):
    def __init__(self, params):
        super().__init__(params)
        self.train_button = QPushButton('Train', self)
        self.train_button.setFixedHeight(self.copy_code_button.height())
        self.train_button.setFont(self.copy_code_button.font())
        self.train_button.setStyleSheet(self.copy_code_button.styleSheet())
        self._layout.insertWidget(0, self.train_button)
        self.train_button.clicked.connect(self.on_train)

    def on_train(self):
        if getattr(self, 'worker', None) is not None and self.worker.isRunning():
            return
        self.node.trained_model = None
        from .utils import clear_outputs
        clear_outputs(self.node)
        self.status_box.setPlainText('Training...')
        self.train_button.setEnabled(False)
        try:
            self._training_revision = self.node.revision
            self.worker = TrainingWorker(self.node.training_inputs())
            self.node.training_worker = self.worker
            self.worker.progress.connect(self.status_box.append)
            self.worker.succeeded.connect(self._completed)
            self.worker.failed.connect(self._failed)
            self.worker.finished.connect(self._finished)
            # Retain workers until completion even if their node widget is removed.
            _training_workers.add(self.worker)
            self.worker.finished.connect(_release_training_workers)
            self.worker.start()
        except Exception as error:
            self._failed(str(error))
            self._finished()

    def _completed(self, model, history):
        if self.node.revision != self._training_revision:
            self.status_box.append('Inputs changed during training; run again to publish results.')
            return
        self.node.trained_model = model
        self.node.set_output_val(0, Data(model))
        self.node.set_output_val(1, Data(history))
        self.status_box.append('Done.')

    def _failed(self, message):
        self.status_box.append(f'Error: {message}')

    def _finished(self):
        self.train_button.setEnabled(True)

@node_gui(train_model_nodes.TrainModelNode)
class TrainModelNodeGui(PyTorchGuiBase):
    main_widget_class = TrainMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'loss': combo_box(
            items=['cross_entropy', 'bce_logits', 'bce', 'mse', 'mae', 'smooth_l1'],
            init_index=0, descr='Loss function',
        ),
        'optimizer': combo_box(
            items=['Adam', 'SGD', 'RMSprop', 'AdamW'],
            init_index=0, descr='Optimizer',
        ),
        'learning_rate': float_spinbox(init=0.001, range=(1e-6, 10.0), decimals=6, step=0.001, descr='Learning rate'),
        'epochs': float_spinbox(init=1, range=(1, 1000), decimals=0, step=1, descr='Number of epochs'),
    }
    init_input_widgets = {
        2: {'name': 'loss',          'pos': 'besides'},
        3: {'name': 'optimizer',     'pos': 'besides'},
        4: {'name': 'learning_rate', 'pos': 'besides'},
        5: {'name': 'epochs',        'pos': 'besides'},
    }


@node_gui(vae.TrainVAENode)
class TrainVAENodeGui(PyTorchGuiBase):
    main_widget_class = ExportTrainingMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'learning_rate': float_spinbox(init=0.001, range=(1e-6, 10.0), decimals=6, step=0.001, descr='Learning rate'),
        'epochs': float_spinbox(init=5, range=(1, 1000), decimals=0, step=1, descr='Number of epochs'),
        'kl_weight': float_spinbox(init=1, range=(0, 100), decimals=2, step=0.1, descr='Weight on KL divergence'),
    }
    init_input_widgets = {
        2: {'name': 'learning_rate', 'pos': 'besides'},
        3: {'name': 'epochs', 'pos': 'besides'},
        4: {'name': 'kl_weight', 'pos': 'besides'},
    }
