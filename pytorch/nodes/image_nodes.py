from dataclasses import dataclass
from functools import partial
from numbers import Integral
from pathlib import Path
from threading import get_ident

from ryven.node_env import *
from ryvencore.Base import Event
from .utils import TransientData, value, boolean, clear_outputs
import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class FolderImages:
    root: Path
    class_names: tuple[str, ...]
    paths: tuple[str, ...]
    labels: tuple[int, ...]


def image_tensor_transform(image_size, normalize):
    steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(steps)


def model_image_transform(model):
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Connect a vision model to use its image preprocessing')
    transform = getattr(model, '_ryven_transform', None)
    return image_tensor_transform(224, True) if transform is None else transform


def rgb_image(source):
    if isinstance(source, Image.Image):
        return source.convert('RGB')
    with Image.open(str(source)) as image:
        return image.convert('RGB')


class ImagesToTensorNode(Node):
    """Convert images or image paths to a batch tensor. Connect a model to use its
    preprocessing, or set the image size and normalization manually.
    """

    title = "Convert Images to Tensor"
    instructions = 'Connect images to see the tensor shape (N, C, H, W).'
    init_inputs = [
        NodeInputType(label="image_paths"),
        NodeInputType(label="image_size", default=Data(224)),
        NodeInputType(label="normalize", default=Data(True)),
        NodeInputType(label="model"),
    ]
    init_outputs = [
        NodeOutputType(label="image_tensor"),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def load_additional_data(self, data):
        if len(self.inputs) == 3:
            self.create_input(label='model')

    def update_event(self, inp=-1):
        clear_outputs(self)
        sources = value(self, 0)
        if sources is None or sources == "":
            self.report_changed.emit(self.instructions)
            return
        folder_images = sources if isinstance(sources, FolderImages) else None
        if folder_images is not None:
            sources = folder_images.paths
        elif not isinstance(sources, (tuple, list)):
            sources = [sources]

        try:
            model = value(self, 3)
            if model is not None:
                transform = model_image_transform(model)
            elif self._inp_connected(3):
                self.report_changed.emit('Load the connected model to use its preprocessing.')
                return
            else:
                image_size = int(value(self, 1, 224))
                if image_size <= 0:
                    raise ValueError('Image size must be positive')
                transform = image_tensor_transform(image_size, boolean(value(self, 2)))
            if sources:
                batch = torch.stack([transform(rgb_image(source)) for source in sources])
                if folder_images is not None:
                    batch._ryven_image_recipe = folder_images, transform
                data_type = TransientData if folder_images is not None else Data
                self.set_output_val(0, data_type(batch))
                preprocessing = 'Model preprocessing' if model is not None else 'Manual preprocessing'
                self.report_changed.emit(f'{tuple(batch.shape)} · {batch.dtype}\n{preprocessing}')
            else:
                self.report_changed.emit(self.instructions)
        except Exception as error:
            self.report_changed.emit(f'Could not convert images: {error}')
            print(f'Convert Images to Tensor: {error}')


class ClassificationOutputNode(Node):
    title = "Classification Output"
    init_inputs = [
        NodeInputType(label="model"),
        NodeInputType(label="image_tensor"),
    ]
    init_outputs = [
        NodeOutputType(label="class_index"),
        NodeOutputType(label="confidence"),
        NodeOutputType(label="probabilities"),
    ]

    def update_event(self, inp=-1):
        model = value(self, 0)
        image_tensor = value(self, 1)
        clear_outputs(self)
        if model is None or image_tensor is None:
            return

        try:
            model.eval()
            tensor = image_tensor
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.float()
            else:
                print("ClassificationOutput: image_tensor is not a torch.Tensor")
                self.set_output_val(0, Data(None))
                self.set_output_val(1, Data(None))
                self.set_output_val(2, Data(None))
                return

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, class_index = torch.max(probs, dim=1)

            self.set_output_val(0, Data(class_index.tolist()))
            self.set_output_val(1, Data(confidence.tolist()))
            self.set_output_val(2, Data(probs))
        except Exception as e:
            print("ClassificationOutput error:", e)
            self.set_output_val(0, Data(None))
            self.set_output_val(1, Data(None))
            self.set_output_val(2, Data(None))


class EvaluateModelNode(Node):
    """Run one image through the model and see its predicted classes. The feature-map
    viewer shows where individual filters in the encoder responded to the same image.
    """

    title = "Predict and View Feature Maps"
    tags = ['evaluate model', 'inference', 'activations']
    PREVIEW_CHANNELS = 8
    init_inputs = [
        NodeInputType(label="model"),
        NodeInputType(label="image_path"),
        NodeInputType(label="expected_class"),
    ]
    init_outputs = []

    def __init__(self, params):
        super().__init__(params)
        self.report = ''
        self.photo_preview = None
        self.feature_preview = None

    def load_additional_data(self, data):
        if len(self.inputs) == 2:
            self.create_input(label='expected_class')

    def update_event(self, inp=-1):
        self.report = ''
        self.photo_preview = None
        self.feature_preview = None

    def request_run(self, *, show_feature_maps=False, image=None):
        self.update_event()
        feature_maps = []
        inference_thread = get_ident()

        def capture(name, _module, _inputs, output):
            if get_ident() == inference_thread:
                # Copy now because a later operation (such as in-place ReLU) can alter the output.
                channels = output[0, :self.PREVIEW_CHANNELS].detach().to(
                    device='cpu', dtype=torch.float32, copy=True,
                )
                feature_maps.append((name, tuple(output.shape[1:]), channels))

        try:
            model = value(self, 0)
            source = value(self, 1) if image is None else image
            if not isinstance(model, torch.nn.Module):
                raise ValueError('Load and connect a model first')
            if isinstance(source, (tuple, list)):
                source = source[0] if source else None
            if not source:
                raise ValueError('Connect a test image or choose an image file')
            photo = rgb_image(source)
            tensor = model_image_transform(model)(photo).unsqueeze(0)
            parameter = next(model.parameters())
            was_training = model.training
            hooks = []
            try:
                if show_feature_maps:
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            hooks.append(module.register_forward_hook(partial(capture, name or 'Conv2d')))
                model.eval()
                with torch.no_grad():
                    probabilities = model(tensor.to(parameter.device, parameter.dtype)).softmax(dim=1)
            finally:
                for hook in hooks:
                    hook.remove()
                model.train(was_training)
            top = probabilities.topk(min(5, probabilities.shape[1]))
            categories = getattr(model, '_ryven_categories', None)
            if categories is not None and len(categories) != probabilities.shape[1]:
                categories = None
            lines = []
            # A wired label belongs to the wired image. An explicitly entered label can test a chosen file.
            expected = None if image is not None and self._inp_connected(2) else value(self, 2)
            if expected is not None:
                if isinstance(expected, str) and categories and expected.strip() in categories:
                    expected = tuple(categories).index(expected.strip())
                if isinstance(expected, Integral) and not isinstance(expected, bool) and 0 <= expected < probabilities.shape[1]:
                    label = categories[expected] if categories else f'class {expected}'
                    outcome = 'Correct prediction' if top.indices[0, 0].item() == expected else 'Incorrect prediction'
                    lines.append(f'Expected {label} · {outcome}')
                else:
                    lines.append(f'Enter a class name or an index from 0 to {probabilities.shape[1] - 1}.')
                lines.append('')

            lines.append('Top predictions:')
            for rank, (index, score) in enumerate(zip(top.indices[0].tolist(), top.values[0].tolist()), 1):
                label = categories[index] if categories else f'class {index}'
                lines.append(f'{rank}. {label} ({score * 100:.1f}%)')
            photo.thumbnail((192, 192), Image.Resampling.LANCZOS)
            self.photo_preview = photo
            if feature_maps:
                self.feature_preview = (photo, feature_maps)
            elif show_feature_maps:
                lines.append('No Conv2d layers ran for this image.')
            self.report = '\n'.join(lines)
        except Exception as error:
            self.report = f'Inference failed: {error}'
