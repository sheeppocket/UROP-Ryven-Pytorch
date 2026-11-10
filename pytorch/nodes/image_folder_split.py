from pathlib import Path
from random import Random

from ryven.node_env import *
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from .image_nodes import FolderImages, model_image_transform
from .utils import TransientData, clear_outputs, value


IMAGE_DATASET_PRESETS = {
    'Bundled digits (0–4)': Path(__file__).resolve().parents[1] / 'demo_data' / 'digits',
}


def stratified_split(dataset):
    by_class = [[] for _ in dataset.classes]
    for index, label in enumerate(dataset.targets):
        by_class[label].append(index)

    rng = Random(42)
    train_indices, test_indices = [], []
    for name, indices in zip(dataset.classes, by_class):
        if len(indices) < 2:
            raise ValueError(f'{name!r} needs at least two images for an automatic split')
        rng.shuffle(indices)
        test_count = max(1, len(indices) // 5)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def folder_splits(folder, test_folder, transform=None):
    if isinstance(folder, str):
        folder = folder.strip()
        if len(folder) >= 2 and folder[0] in ('"', "'") and folder[-1] == folder[0]:
            folder = folder[1:-1]
        preset = IMAGE_DATASET_PRESETS.get(folder)
        if preset is not None:
            folder, test_folder = preset / 'train', preset / 'test'
    if not folder:
        raise ValueError('Enter a folder containing one subfolder per class')

    root = Path(folder).expanduser().resolve()
    train = ImageFolder(str(root), transform=transform)
    if len(train.classes) < 2:
        raise ValueError('Provide at least two class subfolders')
    if not test_folder:
        return stratified_split(train)

    test_root = Path(test_folder).expanduser().resolve()
    if test_root.is_relative_to(root) or root.is_relative_to(test_root):
        raise ValueError('Training and test folders must be separate directories')
    test = ImageFolder(str(test_root), transform=transform)
    if test.class_to_idx != train.class_to_idx:
        raise ValueError('Training and test folders must contain the same class names')
    return train, test


def folder_images(split):
    source = split.dataset if isinstance(split, Subset) else split
    indices = split.indices if isinstance(split, Subset) else range(len(source))
    paths, labels = zip(*(source.samples[index] for index in indices))
    return FolderImages(Path(source.root), tuple(source.classes), paths, labels)


def split_report(names, train, test, label):
    return (f'{", ".join(names)}\n{len(train)} training images · {len(test)} test images\n'
            f'Example label is {label}.')


class ImageFolderPathsNode(Node):
    """Index labeled image folders and split paths and integer labels for training and testing."""

    title = 'Find Labeled Images'
    tags = ['load images', 'image folders', 'train test split']
    instructions = 'Choose a sample or folder to find image paths and class labels.'
    init_inputs = [
        NodeInputType(label='image_folder'),
        NodeInputType(label='test_folder'),
    ]
    init_outputs = [
        NodeOutputType(label='class_names'),
        NodeOutputType(label='train_image_paths'),
        NodeOutputType(label='train_labels'),
        NodeOutputType(label='test_image_paths'),
        NodeOutputType(label='test_labels'),
        NodeOutputType(label='held_out_image'),
        NodeOutputType(label='held_out_label'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.report = self.instructions

    def update_event(self, inp=-1):
        self.report = self.instructions
        clear_outputs(self)
        if inp >= 0:
            return

        try:
            train_split, test_split = folder_splits(value(self, 0), value(self, 1))
            train, test = folder_images(train_split), folder_images(test_split)
            names = train.class_names
            label = names[test.labels[0]]
            outputs = (names, train, torch.tensor(train.labels, dtype=torch.long),
                       test, torch.tensor(test.labels, dtype=torch.long), test.paths[0], label)
            self.report = split_report(names, train.paths, test.paths, label)
            for index, output in enumerate(outputs):
                self.set_output_val(index, TransientData(output))
        except Exception as error:
            clear_outputs(self)
            self.report = f'Could not find labeled images: {error}'


class ImageFolderSplitNode(Node):
    """Choose a bundled dataset or read the class names from your own image subfolders.
    A custom dataset can use a separate test folder or an automatic held-out split.
    """

    title = 'Load Image Datasets (Shortcut)'
    tags = ['image folders', 'labeled dataset']
    instructions = 'Choose a sample or enter a folder, then load the images.'
    init_inputs = [
        NodeInputType(label='model'),
        NodeInputType(label='image_folder'),
        NodeInputType(label='test_folder'),
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
        if len(self.outputs) == 4:
            self.create_output(label='held_out_label')

    def update_event(self, inp=-1):
        self.report = self.instructions
        clear_outputs(self)
        if inp >= 0:
            return

        try:
            model = value(self, 0)
            folder, test_folder = value(self, 1), value(self, 2)
            if not isinstance(model, nn.Module):
                raise ValueError('Load and connect the model first')
            train, test = folder_splits(folder, test_folder, model_image_transform(model))
            if isinstance(test, Subset):
                source = test.dataset
                example_path, example_label = source.samples[test.indices[0]]
            else:
                source = test
                example_path, example_label = source.samples[0]
            names = tuple(source.classes)
            label = names[example_label]
            self.report = split_report(names, train, test, label)
            self.set_output_val(0, Data(names))
            self.set_output_val(1, Data(train))
            self.set_output_val(2, Data(test))
            self.set_output_val(3, Data(example_path))
            self.set_output_val(4, Data(label))
        except Exception as error:
            clear_outputs(self)
            self.report = f'Could not load image split: {error}'
