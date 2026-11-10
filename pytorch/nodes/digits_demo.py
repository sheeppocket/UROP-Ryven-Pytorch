from ryven.node_env import *
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights

from .image_nodes import model_image_transform
from .utils import clear_outputs, value


DEFAULT_DIGITS = (0, 1, 2, 3, 4)
DIGIT_NAMES = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
TRAIN_PER_CLASS = 25
TEST_PER_CLASS = 10


class DigitImages(Dataset):
    def __init__(self, pixels, labels, transform):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def image(self, index):
        return Image.fromarray(self.pixels[index]).resize((96, 96), Image.Resampling.BICUBIC).convert('RGB')

    def __getitem__(self, index):
        return self.transform(self.image(index)), int(self.labels[index])


def digit_image_split(selection, transform):
    if (not isinstance(selection, (tuple, list)) or len(selection) < 2
            or any(type(digit) is not int or not 0 <= digit < len(DIGIT_NAMES) for digit in selection)
            or len(set(selection)) != len(selection)):
        raise ValueError('Choose at least two distinct digits from 0 to 9, such as [0, 1, 2, 3, 4]')

    from sklearn.datasets import load_digits

    digits = load_digits()
    names = tuple(DIGIT_NAMES[digit] for digit in selection)
    labels = np.full(len(DIGIT_NAMES), -1, dtype=int)
    labels[list(selection)] = np.arange(len(selection))
    pixels = (digits.images * (255 / 16)).round().astype(np.uint8)
    rng = np.random.default_rng(42)
    train_indices, test_indices = [], []
    for digit in selection:
        indices = rng.permutation(np.flatnonzero(digits.target == digit))
        train_indices.extend(indices[:TRAIN_PER_CLASS])
        test_indices.extend(indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS])
    train = DigitImages(pixels[train_indices], labels[digits.target[train_indices]], transform)
    train.training_indices = tuple(int(index) for index in train_indices)
    test = DigitImages(pixels[test_indices], labels[digits.target[test_indices]], transform)
    example = (selection.index(3) if 3 in selection else 0) * TEST_PER_CLASS
    return names, train, test, test.image(example), names[test.labels[example]]


class DigitsDemoNode(Node):
    """Pick which handwritten digits to load. The training images teach the
    classifier the chosen labels. The test images are set aside to check predictions
    on examples the classifier has never trained on. Connect the vision model to
    use the same preprocessing for the datasets and the test image prediction.
    """

    title = 'Load Digit Images (Legacy)'
    tags = ['load images', 'handwritten digits']
    legacy_identifiers = ['nodes.Data.DigitsDemoNode']
    instructions = 'Handwritten digits\nChoose the digits, then load the images.'
    init_inputs = [
        NodeInputType(label='digits', default=Data(DEFAULT_DIGITS)),
        NodeInputType(label='model'),
    ]
    init_outputs = [
        NodeOutputType(label='class_names'),
        NodeOutputType(label='train_dataset'),
        NodeOutputType(label='test_dataset'),
        NodeOutputType(label='held_out_image'),
        NodeOutputType(label='held_out_label'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.report = self.instructions

    def load_additional_data(self, data):
        for port in self.init_inputs[len(self.inputs):]:
            self.create_input(label=port.label, default=port.default)
        if len(self.outputs) == 4:
            self.create_output(label='held_out_label')

    def update_event(self, inp=-1):
        if inp >= 0:
            self.report = self.instructions
            clear_outputs(self)
            return
        try:
            model = value(self, 1)
            if model is None and self._inp_connected(1):
                raise ValueError('Load the connected vision model first')
            transform = ResNet18_Weights.DEFAULT.transforms() if model is None else model_image_transform(model)
            split = digit_image_split(value(self, 0), transform)
            names, train, test, _, label = split
            self.report = (f'{", ".join(names)}\n{len(train)} training images · {len(test)} test images\n'
                           f'Example label is {label}.')
            for index, payload in enumerate(split):
                self.set_output_val(index, Data(payload))
        except Exception as error:
            clear_outputs(self)
            self.report = f'Could not load digits: {error}'
