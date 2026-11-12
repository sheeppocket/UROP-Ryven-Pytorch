import ast
from typing import Tuple

from PIL import Image
from qtpy.QtGui import QFont, QImage, QPixmap
from qtpy.QtCore import Qt, Signal, QThread
from qtpy.QtWidgets import (QApplication, QPushButton, QComboBox, QListWidget,
    QListWidgetItem, QHBoxLayout, QDoubleSpinBox, QTextEdit, QWidget,
    QVBoxLayout, QLineEdit, QDialog, QGridLayout, QLabel, QFileDialog, QGroupBox)
from ryvencore import serialize

from ryven.gui_env import *


"""generic base classes"""
class PyTorchGuiBase(NodeGUI):
    color = '#5e548e'

def combo_box(items: list = None, init_index: int = 0, descr: str = '', editable: bool = False):
    """Custom combo box widget with global popup, styled like QComboBox."""
    if items is None:
        items = []
    class StdInpWidget_ComboBox(NodeInputWidget, QComboBox):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QComboBox.__init__(self)
            self._prevent_update = False
            self.setEditable(editable)
            self.addItems(items)
            self.setCurrentIndex(init_index)
            if editable:
                self.setInsertPolicy(QComboBox.NoInsert)
                self.lineEdit().setPlaceholderText('Choose a sample or enter a folder')
            changed = self.editTextChanged if editable else self.currentTextChanged
            changed.connect(self.text_changed)
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
                QComboBox:editable {
                    background: #2c2c2c;
                    color: #ddd;
                }
                QComboBox:editable:focus {
                    border: 1px solid #5e548e;
                    background: #333;
                }
                QComboBox QLineEdit {
                    border: none;
                    background: transparent;
                    color: #ddd;
                    padding: 0px;
                    margin: 0px;
                }
                QComboBox QLineEdit:focus {
                    background: transparent;
                    color: #ddd;
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
            width = self._popup.sizeHintForColumn(0) + 20
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
            if editable:
                self.setEditText(item.text())
            else:
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
                if editable:
                    self.setEditText(str(text) if text is not None else '')
                else:
                    index = self.findText(str(text))
                    if index >= 0:
                        self.setCurrentIndex(index)
                self._prevent_update = False

        def val_update_event(self, val):
            self.set_state({'val': val})

    StdInpWidget_ComboBox.__doc__ = descr
    return StdInpWidget_ComboBox


def composite_fill_input(items: list = None, init_index: int = 0, descr: str = ''):
    """Composite widget: dropdown + text input. If text is non-empty, it takes precedence."""
    if items is None:
        items = []

    class StdInpWidget_CompositeFill(NodeInputWidget, QWidget):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QWidget.__init__(self)

            self._prevent_update = False

            self.layout = QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.layout.setSpacing(4)

            self.combo = QComboBox(self)
            self.combo.addItems(items)
            self.combo.setCurrentIndex(init_index)
            self.combo.setMinimumWidth(120)

            self.line = QLineEdit(self)
            self.line.setPlaceholderText('Type custom fill (overrides dropdown)')
            self.line.setMinimumWidth(120)

            self.layout.addWidget(self.combo)
            self.layout.addWidget(self.line)

            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(9)
            self.combo.setFont(font)
            self.line.setFont(font)

            self.combo.currentIndexChanged.connect(self.combo_changed)
            self.line.textChanged.connect(self.line_changed)

            if descr:
                self.setToolTip(descr)

        def combo_changed(self, idx):
            if not self._prevent_update and not self.line.text().strip():
                self.update_node_input(Data(self.combo.currentText()))

        def line_changed(self, text):
            if not self._prevent_update:
                if text.strip():
                    self.update_node_input(Data(text))
                else:
                    # fall back to combo selection
                    self.update_node_input(Data(self.combo.currentText()))

        @property
        def val(self):
            txt = self.line.text().strip()
            if txt:
                return Data(txt)
            return Data(self.combo.currentText())

        def get_state(self):
            return {'val': self.val.payload}

        def set_state(self, data):
            if 'val' in data:
                self._prevent_update = True
                v = data['val']
                v = v.payload if isinstance(v, Data) else v
                if isinstance(v, str):
                    # prefer text for custom values
                    if v in items:
                        # if matches dropdown, set combo
                        idx = self.combo.findText(v)
                        if idx >= 0:
                            self.combo.setCurrentIndex(idx)
                            self.line.setText('')
                        else:
                            self.line.setText(str(v))
                    else:
                        try:
                            idx = self.combo.findText(str(v))
                            if idx >= 0:
                                self.combo.setCurrentIndex(idx)
                                self.line.setText('')
                            else:
                                self.line.setText(str(v))
                        except Exception:
                            self.line.setText(str(v))
                else:
                    try:
                        self.combo.setCurrentIndex(int(v))
                    except Exception:
                        self.line.setText(str(v))
                self._prevent_update = False

        def val_update_event(self, val):
            self._prevent_update = True
            try:
                payload = val.payload
                if isinstance(payload, str) and payload.strip():
                    # put into line if custom
                    if payload in items:
                        idx = self.combo.findText(payload)
                        if idx >= 0:
                            self.combo.setCurrentIndex(idx)
                            self.line.setText('')
                        else:
                            self.line.setText(payload)
                    else:
                        self.line.setText(payload)
                else:
                    self.line.setText('')
                    try:
                        idx = self.combo.findText(str(payload))
                        if idx >= 0:
                            self.combo.setCurrentIndex(idx)
                    except Exception:
                        pass
            finally:
                self._prevent_update = False

    StdInpWidget_CompositeFill.__doc__ = descr
    return StdInpWidget_CompositeFill

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


def button_update_line_edit(button_text: str = 'Update', init_text: str = '', descr: str = ''):
    """Creates a text input widget with a button that updates its value when pressed."""

    class StdInpWidget_ButtonUpdateLineEdit(NodeInputWidget, QWidget):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QWidget.__init__(self)

            self._prevent_update = False

            self.layout = QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.layout.setSpacing(4)

            self.line = QLineEdit(self)
            self.line.setText(init_text)
            self.line.setMinimumWidth(120)
            self.line.setFont(QFont('Arial', 9))
            self.line.setStyleSheet(
                "QLineEdit {"
                "border: 1px solid #555;"
                "border-radius: 3px;"
                "padding: 2px 5px;"
                "background: #2c2c2c;"
                "color: #ddd;"
                "}"
                "QLineEdit:hover { border: 1px solid #5e548e; }"
                "QLineEdit:focus { border: 1px solid #5e548e; background: #333; }"
            )

            self.button = QPushButton(button_text, self)
            self.button.setFixedHeight(24)
            self.button.setMinimumWidth(70)
            self.button.setFont(QFont('Arial', 9))
            self.button.setStyleSheet(
                "QPushButton { background: #3d3d5c; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 5px; }"
                "QPushButton:hover { border-color: #5e548e; }"
            )

            self.layout.addWidget(self.line)
            self.layout.addWidget(self.button)

            self.line.textChanged.connect(self.text_changed)
            self.button.clicked.connect(self.on_clicked)

            if descr:
                self.setToolTip(descr)

        def text_changed(self, text):
            if not self._prevent_update:
                self.update_node_input(Data(text))

        def on_clicked(self):
            text = 'button pressed'
            self._prevent_update = True
            self.line.setText(text)
            self._prevent_update = False
            self.update_node_input(Data(text))

        @property
        def val(self):
            return Data(self.line.text())

        def get_state(self):
            return {'val': self.line.text()}

        def set_state(self, data):
            if 'val' in data:
                self._prevent_update = True
                self.line.setText(str(data['val']))
                self._prevent_update = False

        def val_update_event(self, val):
            self._prevent_update = True
            self.line.setText(str(val.payload) if val.payload is not None else '')
            self._prevent_update = False

    StdInpWidget_ButtonUpdateLineEdit.__doc__ = descr
    return StdInpWidget_ButtonUpdateLineEdit


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


def add_socket_button():
    """Return a small QPushButton widget class that attempts to add an input socket to its node."""
    class AddSocketWidget(NodeInputWidget, QPushButton):
        def __init__(self, params):
            NodeInputWidget.__init__(self, params)
            QPushButton.__init__(self, 'add input +')
            # wider button to sit flush at top-left
            self.setFixedSize(120, 20)
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(9)
            self.setFont(font)
            # center text and style
            # left-align text and remove heavy rounding so it sits flush
            self.setStyleSheet(
                "QPushButton{ text-align:left; padding-left:8px; color:#ddd; background:#2c2c2c; border:1px solid #555; border-top-left-radius:0px; border-top-right-radius:4px; border-bottom-right-radius:4px; border-bottom-left-radius:0px;}"
                "QPushButton:hover{ border-color:#5e548e}"
            )
            self.clicked.connect(self.on_click)

        def on_click(self):
            self.node.create_input(label=f'tensor {len(self.node.inputs)}')

    AddSocketWidget.__doc__ = "Small '+' button to add inputs to TensorDataset"
    return AddSocketWidget


def photo_pixmap(photo):
    width, height = photo.size
    pixels = photo.tobytes()
    image = QImage(pixels, width, height, width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(image)


class FeatureMapPanel(QGroupBox):
    def __init__(self, title, channel_count, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.layer_name = QLabel(self)
        self.layer_name.setTextFormat(Qt.PlainText)
        layout.addWidget(self.layer_name)
        self.summary = QLabel(self)
        layout.addWidget(self.summary)
        grid = QGridLayout()
        layout.addLayout(grid)
        self.tiles = []
        for index in range(channel_count):
            tile = QVBoxLayout()
            caption = QLabel(self)
            caption.setFixedHeight(18)
            tile.addWidget(caption, alignment=Qt.AlignHCenter)
            preview = QLabel(self)
            preview.setFixedSize(76, 76)
            preview.setAlignment(Qt.AlignCenter)
            tile.addWidget(preview)
            grid.addLayout(tile, index // 4, index % 4)
            self.tiles.append((caption, preview))

    def show_maps(self, layer):
        for caption, preview in self.tiles:
            caption.clear()
            preview.clear()

        name, (count, height, width), channels = layer
        self.layer_name.setText(name)
        self.summary.setText(f'{height} × {width} pixels · Showing {len(channels)} of {count} channels')
        for index, (channel, (caption, preview)) in enumerate(zip(channels, self.tiles), 1):
            channel = channel.nan_to_num(nan=0, posinf=0, neginf=0)
            peak = channel.abs().max().item() or 1.0
            pixels = (128 + 127 * (channel / peak)).round().byte().numpy().tobytes()
            image = QImage(pixels, width, height, width, QImage.Format_Grayscale8)
            preview.setPixmap(QPixmap.fromImage(image).scaled(preview.size(), Qt.KeepAspectRatio))
            caption.setText(f'Channel {index}')


class FeatureMapsDialog(QDialog):
    def __init__(self, photo, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Test image through the encoder')
        self.layers = layers

        layout = QVBoxLayout(self)
        self.reference_picker = QComboBox(self)
        self.reference_picker.setAccessibleName('Reference layer')
        self.picker = QComboBox(self)
        self.picker.setAccessibleName('Comparison layer')
        self.previous_button = QPushButton('Previous', self)
        self.next_button = QPushButton('Next', self)
        self.progress = QLabel(self)
        choices = [f'{index}. {name}' for index, (name, _, _) in enumerate(layers, 1)]
        self.reference_picker.addItems(choices)
        self.picker.addItems(choices)
        self.picker.setCurrentIndex(len(layers) - 1)

        navigation = QHBoxLayout()
        navigation.addWidget(QLabel('Reference layer', self))
        navigation.addWidget(self.reference_picker, stretch=1)
        navigation.addWidget(QLabel('Comparison layer', self))
        navigation.addWidget(self.picker, stretch=1)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.progress)
        layout.addLayout(navigation)

        content = QHBoxLayout()
        layout.addLayout(content)
        original = QVBoxLayout()
        original.addWidget(QLabel('Test image', self), alignment=Qt.AlignHCenter)
        preview = QLabel(self)
        preview.setPixmap(photo_pixmap(photo).scaled(144, 144, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        original.addWidget(preview, alignment=Qt.AlignHCenter)
        content.addLayout(original)
        content.setAlignment(original, Qt.AlignVCenter)

        channel_count = max(len(channels) for _, _, channels in layers)
        self.reference_layer = FeatureMapPanel('Reference convolution', channel_count, self)
        self.selected_layer = FeatureMapPanel('Comparison convolution', channel_count, self)
        content.addWidget(self.reference_layer)
        content.addWidget(self.selected_layer)

        legend = QLabel('Layers are shown in the order they ran. These are raw convolution outputs. '
                        'Dark is negative, gray is zero, and light is positive. Each channel is scaled separately.', self)
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.previous_button.clicked.connect(lambda: self.picker.setCurrentIndex(self.picker.currentIndex() - 1))
        self.next_button.clicked.connect(lambda: self.picker.setCurrentIndex(self.picker.currentIndex() + 1))
        self.reference_picker.currentIndexChanged.connect(self.show_reference_layer)
        self.picker.currentIndexChanged.connect(self.show_layer)
        self.show_reference_layer(self.reference_picker.currentIndex())
        self.show_layer(self.picker.currentIndex())

    def show_reference_layer(self, index):
        self.reference_layer.show_maps(self.layers[index])

    def show_layer(self, index):
        self.progress.setText(f'Layer {index + 1} of {len(self.layers)}')
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.layers) - 1)
        self.selected_layer.show_maps(self.layers[index])


class EvaluateModelMainWidget(NodeMainWidget, QWidget):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QWidget.__init__(self)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.run_button = QPushButton('Test Image', self)
        self.run_button.setFixedHeight(26)
        self.run_button.setFont(QFont('Arial', 9))
        self.run_button.setStyleSheet(
            "QPushButton{background:#3d3d5c;color:#ddd;border:1px solid #555;border-radius:3px;}"
            "QPushButton:hover{border-color:#5e548e;}"
        )
        self.feature_button = QPushButton('View Feature Maps', self)
        self.feature_button.setFixedHeight(self.run_button.height())
        self.feature_button.setFont(self.run_button.font())
        self.feature_button.setStyleSheet(self.run_button.styleSheet())
        self.file_button = QPushButton('Inspect Image File…', self)
        self.file_button.setFixedHeight(self.run_button.height())
        self.file_button.setFont(self.run_button.font())
        self.file_button.setStyleSheet(self.run_button.styleSheet())
        self.file_button.setToolTip('Choose one image and open its feature maps')
        self.report_box = QTextEdit(self)
        self.report_box.setReadOnly(True)
        self.report_box.setMinimumHeight(120)
        self.report_box.setStyleSheet(
            "QTextEdit{border:1px solid #555;border-radius:3px;padding:4px;background:#1e1e1e;color:#ddd;}"
        )
        self.image_preview = QLabel(self)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setFixedHeight(108)
        self.image_preview.hide()
        self._layout.addWidget(self.run_button)
        self._layout.addWidget(self.feature_button)
        self._layout.addWidget(self.file_button)
        self._layout.addWidget(self.image_preview)
        self._layout.addWidget(self.report_box)
        self.run_button.clicked.connect(self.on_run)
        self.feature_button.clicked.connect(self.on_view_features)
        self.file_button.clicked.connect(self.on_inspect_file)
        self.node_gui.updating.connect(self.clear_result)

    def clear_result(self):
        self.report_box.clear()
        if not self.image_preview.isHidden():
            self.image_preview.clear()
            self.image_preview.hide()
            self.update_node_shape()

    def show_result(self):
        self.clear_result()
        self.report_box.setPlainText(self.node.report)
        if self.node.photo_preview is not None:
            preview = photo_pixmap(self.node.photo_preview).scaled(160, 108, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview.setPixmap(preview)
            self.image_preview.show()
            self.update_node_shape()

    def on_run(self):
        self.node.request_run()
        self.show_result()

    def on_view_features(self):
        self.inspect_layers()

    def on_inspect_file(self):
        path, _ = QFileDialog.getOpenFileName(self.node_gui.flow_view().window(), 'Choose a test image', '',
                                              'Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)')
        if path:
            self.inspect_layers(image=path)

    def inspect_layers(self, image=None):
        self.node.request_run(show_feature_maps=True, image=image)
        self.show_result()
        if self.node.feature_preview is not None:
            photo, layers = self.node.feature_preview
            dialog = FeatureMapsDialog(photo, layers, self.node_gui.flow_view().window())
            dialog.exec_()
            dialog.deleteLater()

    def get_state(self):
        return {'report': self.report_box.toPlainText()}

    def set_state(self, data):
        if 'report' in data:
            self.report_box.setPlainText(str(data['report']))



# The following GUI classes are specific to certain nodes and use the above widget factories for their inputs

#torch_utils_data --------------------------------------------------------------------------------------------------
from . import torch_utils_data as torch_utils_data_nodes
from . import image_nodes as image_nodes_nodes
from . import digits_demo as digits_demo_nodes
from . import image_folder_split as image_folder_split_nodes
from . import learning_samples
from . import dense_layers
from . import frozen_classifier as frozen_classifier_nodes
from . import test_model as test_model_nodes


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


@node_gui(torch_utils_data_nodes.MakeTensorNode)
class MakeTensorNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'shape': evaled_line_edit('(3,3)', 'Tensor shape as tuple'),
        'dtype': combo_box(
            items=[
                'float32',
                'float64',
                'float16',
                'int32',
                'int64',
                'bool',
            ],
            init_index=0,
            descr='Tensor dtype',
            editable=False
        ),
        'fill': composite_fill_input(
            items=[
                'randn',
                'rand',
                'zeros',
                'ones',
            ],
            init_index=0,
            descr='Tensor fill type'
        ),
    }
    init_input_widgets = {
        0: {'name': 'shape', 'pos': 'besides'},
        1: {'name': 'dtype', 'pos': 'besides'},
        2: {'name': 'fill', 'pos': 'besides'},
    }

@node_gui(torch_utils_data_nodes.TensorDatasetNode)
class TensorDatasetNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        '': add_socket_button(),
    }
    init_input_widgets = {
        0: {'name': '', 'pos': 'besides'},
    }
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
            editable=False
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


@node_gui(image_nodes_nodes.ImagesToTensorNode)
class ImagesToTensorNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    input_widget_classes = {
        'image_paths': evaled_line_edit('', "Image path or list of paths (e.g. ['img1.png', 'img2.png'])"),
        'image_size': float_spinbox(init=224, range=(1, 4096), decimals=0, step=1, descr='Output image size'),
        'normalize': combo_box(
            items=['True', 'False'],
            init_index=0,
            descr='Normalize image tensors',
            editable=False
        ),
    }
    init_input_widgets = {
        0: {'name': 'image_paths', 'pos': 'besides'},
        1: {'name': 'image_size', 'pos': 'besides'},
        2: {'name': 'normalize', 'pos': 'besides'},
    }

    def __init__(self, params):
        super().__init__(params)
        if self.node.load_data is not None:
            self.node.load_data.setdefault('main widget data', serialize({}))


class ImagePathWidget(evaled_line_edit('', 'Enter an image path or connect an image')):
    def val_update_event(self, val):
        if val is not None and isinstance(val.payload, Image.Image):
            val = Data('Connected image')
        super().val_update_event(val)


@node_gui(image_nodes_nodes.EvaluateModelNode)
class EvaluateModelNodeGui(PyTorchGuiBase):
    main_widget_class = EvaluateModelMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'image_path': ImagePathWidget,
        'expected_class': evaled_line_edit('', 'Optional true class name or zero-based class index'),
    }
    init_input_widgets = {
        1: {'name': 'image_path', 'pos': 'besides'},
        2: {'name': 'expected_class', 'pos': 'besides'},
    }


class DatasetSplitMainWidget(NodeMainWidget, QWidget):
    button_text = 'Load Images'

    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QWidget.__init__(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.load_button = QPushButton(self.button_text, self)
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


@node_gui(digits_demo_nodes.DigitsDemoNode)
class DigitsDemoNodeGui(PyTorchGuiBase):
    main_widget_class = DatasetSplitMainWidget
    input_widget_classes = {
        'digits': evaled_line_edit(str(list(digits_demo_nodes.DEFAULT_DIGITS)), 'Choose two or more digits from 0 to 9'),
    }
    init_input_widgets = {0: {'name': 'digits', 'pos': 'besides'}}


@node_gui(image_folder_split_nodes.ImageFolderSplitNode)
class ImageFolderSplitNodeGui(PyTorchGuiBase):
    main_widget_class = DatasetSplitMainWidget
    input_widget_classes = {
        'image_folder': combo_box(
            list(image_folder_split_nodes.IMAGE_DATASET_PRESETS), init_index=-1, editable=True,
            descr='Choose a bundled dataset or enter a folder with one subfolder per class.',
        ),
        'test_folder': evaled_line_edit('', 'Optional separate test folder with the same class subfolders'),
    }
    init_input_widgets = {
        1: {'name': 'image_folder', 'pos': 'besides'},
        2: {'name': 'test_folder', 'pos': 'besides'},
    }


class FindImagesMainWidget(DatasetSplitMainWidget):
    button_text = 'Find Images'


@node_gui(image_folder_split_nodes.ImageFolderPathsNode)
class ImageFolderPathsNodeGui(ImageFolderSplitNodeGui):
    main_widget_class = FindImagesMainWidget
    init_input_widgets = {
        0: {'name': 'image_folder', 'pos': 'besides'},
        1: {'name': 'test_folder', 'pos': 'besides'},
    }


class SampleDataMainWidget(DatasetSplitMainWidget):
    button_text = 'Load Examples'


@node_gui(learning_samples.SampleDatasetNode)
class SampleDatasetNodeGui(PyTorchGuiBase):
    main_widget_class = SampleDataMainWidget
    input_widget_classes = {
        'sample': combo_box(list(learning_samples.SampleDatasetNode.options), descr='Sample dataset'),
    }
    init_input_widgets = {0: {'name': 'sample', 'pos': 'besides'}}


class SampleModelMainWidget(DatasetSplitMainWidget):
    button_text = 'Build / Reset Model'


@node_gui(learning_samples.SampleModelNode)
class SampleModelNodeGui(PyTorchGuiBase):
    main_widget_class = SampleModelMainWidget
    input_widget_classes = {
        'architecture': combo_box(list(learning_samples.SampleModelNode.options), descr='Sample model architecture'),
    }
    init_input_widgets = {0: {'name': 'architecture', 'pos': 'besides'}}


@node_gui(frozen_classifier_nodes.FrozenResNetClassifierNode)
class FrozenResNetClassifierNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    input_widget_classes = {
        'class_names': evaled_line_edit('', "Class names in label order, e.g. ['zero', 'one']"),
        'trained_weights': evaled_line_edit('', 'Optional training_result.pth from exported classifier training'),
    }
    init_input_widgets = {
        1: {'name': 'class_names', 'pos': 'besides'},
        2: {'name': 'trained_weights', 'pos': 'besides'},
    }

    def __init__(self, params):
        super().__init__(params)
        if self.node.load_data is not None:
            # Older projects did not have a report widget on this node.
            self.node.load_data.setdefault('main widget data', serialize({}))


@node_gui(test_model_nodes.TestModelNode)
class TestModelNodeGui(PyTorchGuiBase):
    main_widget_class = NodeReportMainWidget
    main_widget_pos = 'between ports'


#weights --------------------------------------------------------------------------------------------------
from . import weights as weights_nodes

@node_gui(weights_nodes.PresetWeightsNode)
class PresetWeightsNodeGui(PyTorchGuiBase):
    input_widget_classes = {
        'model_type': combo_box(
            items=weights_nodes.PresetWeightsNode.MODEL_OPTIONS,
            init_index=0,
            descr='Model type',
            editable=False
        ),
    }
    init_input_widgets = {
        0: {'name': 'model_type', 'pos': 'besides'},
    }


# models --------------------------------------------------------------------------------------------------
from . import models as models_nodes


class LoadModelMainWidget(NodeMainWidget, QPushButton):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QPushButton.__init__(self, 'Load Model')
        self.setFixedHeight(26)
        self.setFont(QFont('Arial', 9))
        self.setStyleSheet(
            "QPushButton{background:#3d3d5c;color:#ddd;border:1px solid #555;border-radius:3px;padding:2px 8px;}"
            "QPushButton:hover{border-color:#5e548e;}"
            "QPushButton:pressed{background:#5e548e;}"
        )
        self.clicked.connect(self.update_node)


@node_gui(models_nodes.ModelsNode)
class ModelsNodeGui(PyTorchGuiBase):
    main_widget_class = LoadModelMainWidget
    main_widget_pos = 'between ports'
    input_widget_classes = {
        'model_type': combo_box(
            items=models_nodes.ModelsNode.MODEL_OPTIONS,
            init_index=models_nodes.ModelsNode.MODEL_OPTIONS.index('resnet18'),
            descr='Model type',
            editable=False
        ),
        'weights': evaled_line_edit('', 'Optional torchvision weights enum or string'),
    }
    init_input_widgets = {
        0: {'name': 'model_type', 'pos': 'besides'},
        1: {'name': 'weights', 'pos': 'besides'},
    }


#train_model --------------------------------------------------------------------------------------------------
from . import train_model as train_model_nodes


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


class TrainMainWidget(NodeMainWidget, QWidget):
    def __init__(self, params):
        NodeMainWidget.__init__(self, params)
        QWidget.__init__(self)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self.train_button = QPushButton('Train', self)
        self.train_button.setFixedHeight(26)
        self.train_button.setFont(QFont('Arial', 9))
        self.train_button.setStyleSheet(
            "QPushButton{background:#3d3d5c;color:#ddd;border:1px solid #555;border-radius:3px;}"
            "QPushButton:hover{border-color:#5e548e;}"
            "QPushButton:pressed{background:#5e548e;}"
        )
        self.copy_code_button = QPushButton('Copy Training Code', self)
        self.copy_code_button.setFixedHeight(self.train_button.height())
        self.copy_code_button.setFont(self.train_button.font())
        self.copy_code_button.setStyleSheet(self.train_button.styleSheet())
        self.status_box = QTextEdit(self)
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(80)
        self.status_box.setStyleSheet(
            "QTextEdit{border:1px solid #555;border-radius:3px;padding:4px;background:#1e1e1e;color:#ddd;}"
        )
        self._layout.addWidget(self.train_button)
        self._layout.addWidget(self.copy_code_button)
        self._layout.addWidget(self.status_box)
        self.train_button.clicked.connect(self.on_train)
        self.copy_code_button.clicked.connect(self.on_copy_training_code)

    def on_copy_training_code(self):
        try:
            code = self.node.export_training_code()
        except Exception as error:
            self.status_box.append(f'Could not copy training code. {error}')
            return
        QApplication.clipboard().setText(code)
        self.status_box.append('Training code copied to the clipboard.')

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

    def get_state(self):
        return {'status': self.status_box.toPlainText()}

    def set_state(self, data):
        if 'status' in data:
            self.status_box.setPlainText(str(data['status']))


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
