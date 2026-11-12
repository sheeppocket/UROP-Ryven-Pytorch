from inspect import getsource
from math import isfinite
from pprint import pformat
from textwrap import dedent

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, default_collate

from .dense_layers import DenseModelSpec
from .learning_samples import LINEAR_DATA, XOR_DATA, sample_split
from .torch_utils_data import RemoteBatches, RemoteCsvDataset, RemoteImageFolder
from .train_model import LOSS_FUNCTIONS, train_model, validate_dense_model


def training_code(model, loader, loss, optimizer, learning_rate, epochs):
    if not isinstance(model, DenseModelSpec):
        raise ValueError('Connect the last Dense Layer to Train Model.')
    validate_dense_model(model, loss)
    remote_batches = isinstance(loader, RemoteBatches)
    if not remote_batches:
        if not isinstance(loader, DataLoader):
            raise ValueError('Connect the training batches before copying training code.')
        if (type(loader.sampler) not in (RandomSampler, SequentialSampler) or loader.batch_size is None
                or loader.collate_fn is not default_collate or getattr(loader.sampler, 'replacement', False)):
            raise ValueError('Use Group Examples into Batches for the portable training loader.')
    if loss not in LOSS_FUNCTIONS or optimizer not in ('Adam', 'SGD', 'RMSprop', 'AdamW'):
        raise ValueError('Choose a supported loss and optimizer')
    if not isfinite(learning_rate) or learning_rate <= 0 or epochs < 1:
        raise ValueError('Learning rate and epochs must be positive')

    settings = {
        'batch_size': loader.batch_size,
        'shuffle': loader.shuffle if remote_batches else isinstance(loader.sampler, RandomSampler),
        'drop_last': False if remote_batches else loader.drop_last,
        'loss': loss,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'seed': 42,
        'output_checkpoint': 'training_result.pth',
    }
    return dense_training_code(model, loader, settings)


def dense_training_code(model, loader, settings):
    imports = 'import torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader'
    widths = f'INPUT_FEATURES = {model.input_features}\nOUTPUT_NEURONS = {model.layers[-1][0]}'
    encoder = model.encoder
    if encoder:
        if not isinstance(loader, RemoteBatches) or not isinstance(loader.dataset, RemoteImageFolder):
            raise ValueError('Connect Image Folder for Remote Training through Group Examples into Batches.')
        if settings['loss'] != 'cross_entropy':
            raise ValueError('Image folder classification uses cross entropy.')
        imports += '\nfrom torchvision import models\nfrom torchvision.datasets import ImageFolder'
        data_config = (
            f'DATA_PATH = {loader.dataset.path!r}\n'
            f'ENCODER_WEIGHTS = models.get_model_weights(models.{encoder.architecture}).DEFAULT'
        )
        dataset_code = ENCODER_IMAGE_DATASET
    elif isinstance(loader, RemoteBatches):
        if not isinstance(loader.dataset, RemoteCsvDataset):
            raise ValueError('Use a CSV for dense training, or connect Image Encoder to use an image folder.')
        imports = 'import csv\nfrom array import array\n\n' + imports
        imports += ', TensorDataset'
        data_config = f'DATA_PATH = {loader.dataset.path!r}'
        dataset_code = DENSE_CSV_DATASET
    else:
        imports += ', TensorDataset'
        dataset = loader.dataset
        sample = getattr(dataset, '_ryven_sample', None)
        if sample not in (LINEAR_DATA, XOR_DATA):
            raise ValueError('For dense export, connect a remote CSV or the Linear boundary or XOR sample.')
        if model.input_features != 2 or settings['loss'] != 'cross_entropy':
            raise ValueError('The linear and XOR samples require two input features and cross entropy.')
        data_config = (
            f'SAMPLE = {sample!r}\nLINEAR_DATA = {LINEAR_DATA!r}\n'
            f'TRAINING_EXAMPLES = {len(dataset)}'
        )
        dataset_code = getsource(sample_split) + '\n\n' + dedent('''\
            def make_training_dataset():
                generator = torch.Generator().manual_seed(42)
                return sample_split(SAMPLE, TRAINING_EXAMPLES, generator)
        ''')

    model_setup = ['def make_model():']
    modules = []
    container = 'nn.Sequential'
    if encoder:
        weights = 'ENCODER_WEIGHTS' if encoder.weights == 'DEFAULT' else 'None'
        model_setup.extend((
            f'    encoder = models.{encoder.architecture}(weights={weights})',
            '    encoder.fc = nn.Identity()',
        ))
        modules.append('        encoder,')
        if encoder.frozen:
            model_setup.append('    encoder.requires_grad_(False).eval()')
            container = 'FrozenEncoderModel'
    modules.extend(dense_module_lines(model, '        '))
    model_code = '\n'.join((*model_setup, f'    return {container}(', *modules, '    )'))
    if encoder and encoder.frozen:
        model_code = FROZEN_ENCODER_MODEL + '\n\n' + model_code
    selected_loss = settings['loss']
    config = (
        f'{widths}\n{data_config}\n\nSETTINGS = {pformat(settings, sort_dicts=False)}\n'
        f'LOSS_FUNCTIONS = {{{selected_loss!r}: nn.{LOSS_FUNCTIONS[selected_loss].__name__}}}'
    )
    requirements = 'PyTorch and torchvision' if encoder else 'PyTorch'
    return '\n\n'.join((
        f'# Paste into a Colab cell or run as a Python script. Requires {requirements}.',
        imports, config, model_code, dataset_code, getsource(train_model), DENSE_SCRIPT_MAIN,
    ))


def dense_module_lines(model, indentation):
    for inputs, neurons, activation in model.linear_layers():
        yield f'{indentation}nn.Linear({inputs}, {neurons}),'
        if activation != 'None':
            arguments = 'dim=-1' if activation == 'Softmax' else ''
            yield f'{indentation}nn.{activation}({arguments}),'


FROZEN_ENCODER_MODEL = dedent('''\
    class FrozenEncoderModel(nn.Sequential):
        def train(self, mode=True):
            super().train(mode)
            self[0].eval()
            return self
''')


ENCODER_IMAGE_DATASET = dedent('''\
    def make_training_dataset():
        dataset = ImageFolder(DATA_PATH, transform=ENCODER_WEIGHTS.transforms())
        if len(dataset.classes) != OUTPUT_NEURONS:
            raise ValueError(
                f'The last Dense Layer has {OUTPUT_NEURONS} neurons, but {DATA_PATH} '
                f'has {len(dataset.classes)} class folders: {dataset.classes}'
            )
        return dataset
''')


DENSE_CSV_DATASET = dedent('''\
    def make_training_dataset():
        classification = SETTINGS['loss'] == 'cross_entropy'
        target_columns = 1 if classification else OUTPUT_NEURONS
        columns = INPUT_FEATURES + target_columns
        inputs = array('f')
        targets = array('q' if classification else 'f')
        with open(DATA_PATH, newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            header = next(reader, [])
            if len(header) != columns:
                raise ValueError(f'Expected a header with {columns} columns in {DATA_PATH}')
            for line, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != columns:
                    raise ValueError(f'CSV row {line}: expected {columns} columns')
                try:
                    features = [float(cell) for cell in row[:INPUT_FEATURES]]
                    if classification:
                        label = int(row[-1])
                        if not 0 <= label < OUTPUT_NEURONS:
                            raise ValueError(f'class index must be in [0, {OUTPUT_NEURONS - 1}]')
                        targets.append(label)
                    else:
                        targets.extend(float(cell) for cell in row[INPUT_FEATURES:])
                except ValueError as error:
                    raise ValueError(f'CSV row {line}: {error}') from error
                inputs.extend(features)
        if not inputs:
            raise ValueError(f'No training examples in {DATA_PATH}')
        target_dtype = torch.long if classification else torch.float32
        feature_tensor = torch.frombuffer(inputs, dtype=torch.float32).reshape(-1, INPUT_FEATURES)
        target_tensor = torch.frombuffer(targets, dtype=target_dtype)
        if not classification:
            target_tensor = target_tensor.reshape(-1, OUTPUT_NEURONS)
        return TensorDataset(feature_tensor, target_tensor)
''')


DENSE_SCRIPT_MAIN = dedent('''\
    def main():
        torch.manual_seed(SETTINGS['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = make_training_dataset()
        model = make_model().to(device)
        loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], shuffle=SETTINGS['shuffle'],
                            drop_last=SETTINGS['drop_last'])
        print(f'Training on {device} with {len(dataset)} examples', flush=True)
        model, history = train_model(
            model, loader, SETTINGS['loss'], SETTINGS['optimizer'],
            SETTINGS['learning_rate'], SETTINGS['epochs'],
            progress=lambda epoch, total, loss: print(f'Epoch {epoch}/{total} loss={loss:.4f}', flush=True),
        )
        output = SETTINGS['output_checkpoint']
        checkpoint = {'model_state_dict': model.cpu().state_dict(), 'loss_history': history}
        if hasattr(dataset, 'classes'):
            checkpoint['class_names'] = dataset.classes
        torch.save(checkpoint, output)
        print(f'Saved {output}', flush=True)


    if __name__ == '__main__':
        main()
''')
