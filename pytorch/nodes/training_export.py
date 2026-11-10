from inspect import getsource
from math import isfinite
from pathlib import Path
from pprint import pformat
from textwrap import dedent

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, TensorDataset, default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms._presets import ImageClassification

from .digits_demo import DigitImages
from .frozen_classifier import FrozenVisionClassifier, find_classifier
from .image_nodes import image_tensor_transform
from .models import MODEL_CONSTRUCTORS
from .train_model import LOSS_FUNCTIONS, train_model


def describe_preprocessing(transform):
    if isinstance(transform, ImageClassification):
        return {
            'resize_size': transform.resize_size,
            'crop_size': transform.crop_size,
            'interpolation': transform.interpolation.name,
            'antialias': transform.antialias,
            'mean': transform.mean,
            'std': transform.std,
        }
    if isinstance(transform, transforms.Compose) and transform.transforms:
        resize = transform.transforms[0]
        if isinstance(resize, transforms.Resize) and isinstance(resize.size, (tuple, list)):
            image_size, normalize = resize.size[0], len(transform.transforms) == 3
            if repr(transform) == repr(image_tensor_transform(image_size, normalize)):
                return {'image_size': image_size, 'normalize': normalize}
    raise ValueError('Use Convert Images to Tensor with supported model or manual preprocessing')


def describe_image_folder(root, class_names, paths, transform):
    return {
        'source': 'image_folder',
        'root': root.name,
        'class_names': list(class_names),
        'training_files': [Path(path).relative_to(root).as_posix() for path in paths],
        'preprocessing': describe_preprocessing(transform),
    }


def describe_dataset(dataset):
    if isinstance(dataset, DigitImages) and hasattr(dataset, 'training_indices'):
        return {
            'source': 'digits',
            'indices': list(dataset.training_indices),
            'labels': dataset.labels.tolist(),
            'preprocessing': describe_preprocessing(dataset.transform),
        }

    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    if isinstance(source, ImageFolder):
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(source))
        paths = (source.samples[index][0] for index in indices)
        return describe_image_folder(Path(source.root), source.classes, paths, source.transform)

    if isinstance(dataset, TensorDataset) and len(dataset.tensors) == 2:
        images, labels = dataset.tensors
        recipe = getattr(images, '_ryven_image_recipe', None)
        if recipe is not None:
            folder, transform = recipe
            if labels.dtype != torch.long or labels.tolist() != list(folder.labels):
                raise ValueError('Connect the label tensor from the same image split to export training code')
            return describe_image_folder(folder.root, folder.class_names, folder.paths, transform)

    raise ValueError('Connect training data from Find Labeled Images and Convert Images to Tensor to export it')


def training_code(model, loader, loss, optimizer, learning_rate, epochs):
    spec = getattr(model, '_ryven_model_spec', None)
    if spec is None or spec[0] not in MODEL_CONSTRUCTORS:
        raise ValueError('Load a backbone with Load Vision Model before copying training code')
    if not isinstance(loader, DataLoader):
        raise ValueError('Connect the training batches before copying training code')
    if (type(loader.sampler) not in (RandomSampler, SequentialSampler) or loader.batch_size is None
            or loader.collate_fn is not default_collate or getattr(loader.sampler, 'replacement', False)):
        raise ValueError('Use Group Examples into Batches for the portable training loader')
    if loss not in LOSS_FUNCTIONS or optimizer not in ('Adam', 'SGD', 'RMSprop', 'AdamW'):
        raise ValueError('Choose a supported loss and optimizer')
    if not isfinite(learning_rate) or learning_rate <= 0 or epochs < 1:
        raise ValueError('Learning rate and epochs must be positive')

    architecture, weights, initial_checkpoint = spec
    frozen = isinstance(model, FrozenVisionClassifier)
    settings = {
        'architecture': architecture,
        'weights': weights,
        'initial_checkpoint': Path(initial_checkpoint).name if initial_checkpoint else None,
        'frozen_backbone': frozen,
        'class_names': model._ryven_categories if frozen else None,
        'batch_size': loader.batch_size,
        'shuffle': isinstance(loader.sampler, RandomSampler),
        'drop_last': loader.drop_last,
        'loss': loss,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'seed': 42,
        'output_checkpoint': 'training_result.pth',
    }
    dataset = describe_dataset(loader.dataset)

    imports = dedent('''\
        # Ryven training export. Paste this into a Colab cell or run it as a Python script.
        # Requires PyTorch and torchvision.
        # For image folders, update DATA_ROOT after moving the dataset.
        # For a custom starting checkpoint, place that file beside the script or update its path.
        from copy import deepcopy
        from pathlib import Path

        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    ''')
    dataset_helpers = ()
    if dataset['source'] == 'digits':
        imports += 'import numpy as np\nfrom PIL import Image\nfrom torch.utils.data import Dataset\n'
        dataset_helpers = (DigitImages,)
        dataset_code = LEGACY_DIGIT_DATASET
    else:
        imports += 'from torch.utils.data import Subset\nfrom torchvision.datasets import ImageFolder\n'
        dataset_code = FOLDER_DATASET
    loss_entries = '\n'.join(f'    {key!r}: nn.{cls.__name__},' for key, cls in LOSS_FUNCTIONS.items())
    configuration = (
        f'SETTINGS = {pformat(settings, sort_dicts=False)}\n'
        f'DATA = {pformat(dataset, sort_dicts=False)}\n'
        'DATA_ROOT = Path(DATA.get("root", "."))\n\n'
        f'LOSS_FUNCTIONS = {{\n{loss_entries}\n}}\n'
    )
    helpers = '\n\n'.join(getsource(item) for item in (
        image_tensor_transform, *dataset_helpers, find_classifier, FrozenVisionClassifier, train_model,
    ))
    return '\n\n'.join((imports, configuration, helpers, dataset_code, SCRIPT_MAIN))


LEGACY_DIGIT_DATASET = dedent('''\
    def make_training_dataset():
        from sklearn.datasets import load_digits

        digits = load_digits()
        pixels = (digits.images * (255 / 16)).round().astype(np.uint8)
        return DigitImages(pixels[DATA['indices']], np.asarray(DATA['labels']), make_preprocessing())
''')


FOLDER_DATASET = dedent('''\
    def make_training_dataset():
        images = ImageFolder(str(DATA_ROOT), transform=make_preprocessing())
        if images.classes != DATA['class_names']:
            raise ValueError('The class folders must match the original training dataset')
        indices = {Path(path).relative_to(DATA_ROOT).as_posix(): index
                   for index, (path, _) in enumerate(images.samples)}
        return Subset(images, [indices[path] for path in DATA['training_files']])
''')


SCRIPT_MAIN = dedent('''\
    def make_preprocessing():
        config = DATA['preprocessing']
        if 'image_size' in config:
            return image_tensor_transform(config['image_size'], config['normalize'])
        return transforms.Compose([
            transforms.Resize(config['resize_size'],
                              interpolation=transforms.InterpolationMode[config['interpolation']],
                              antialias=config['antialias']),
            transforms.CenterCrop(config['crop_size']),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(config['mean'], config['std']),
        ])


    def main():
        torch.manual_seed(SETTINGS['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        constructor = getattr(models, SETTINGS['architecture'])
        weights_name = SETTINGS['weights']
        weights = models.get_model_weights(constructor)[weights_name] if weights_name else None
        model = constructor(weights=weights)
        if SETTINGS['initial_checkpoint']:
            state = torch.load(SETTINGS['initial_checkpoint'], map_location='cpu', weights_only=True)
            model.load_state_dict(state.get('model_state_dict', state))
        if SETTINGS['frozen_backbone']:
            model = FrozenVisionClassifier(model, SETTINGS['class_names'])
        model = model.to(device)

        dataset = make_training_dataset()
        loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], shuffle=SETTINGS['shuffle'],
                            drop_last=SETTINGS['drop_last'])
        print(f'Training on {device} with {len(dataset)} examples', flush=True)
        model, history = train_model(
            model, loader, SETTINGS['loss'], SETTINGS['optimizer'],
            SETTINGS['learning_rate'], SETTINGS['epochs'],
            progress=lambda epoch, total, loss: print(f'Epoch {epoch}/{total} loss={loss:.4f}', flush=True),
        )

        checkpoint = {
            'architecture': SETTINGS['architecture'],
            'weights': weights_name,
            'frozen_backbone': SETTINGS['frozen_backbone'],
            'class_names': SETTINGS['class_names'],
            'loss_history': history,
        }
        if SETTINGS['frozen_backbone'] and weights_name:
            checkpoint['classifier_state_dict'] = model.classifier.cpu().state_dict()
        else:
            checkpoint['model_state_dict'] = model.cpu().state_dict()
        output = SETTINGS['output_checkpoint']
        torch.save(checkpoint, output)
        print(f'Saved {output}', flush=True)


    if __name__ == '__main__':
        main()
''')
