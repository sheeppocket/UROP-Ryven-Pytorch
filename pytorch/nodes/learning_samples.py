import torch
from torch import nn
from torch.utils.data import TensorDataset

from ryven.node_env import Data, Node, NodeInputType, NodeOutputType

from .utils import clear_outputs, value


LINEAR_DATA = 'Linear boundary'
XOR_DATA = 'XOR quadrants'
SEQUENCE_DATA = 'Sequence order'
LINEAR_MODEL = 'Linear classifier'
MLP_MODEL = 'Two-layer neural network'
RNN_MODEL = 'RNN + classifier'


def sample_split(sample, count, generator):
    if sample == SEQUENCE_DATA:
        # Each pair has identical values but the two markers appear in opposite order.
        noise = torch.rand(count // 2, 6, 1, generator=generator) * 0.5 - 0.25
        inputs = noise.repeat(2, 1, 1)
        labels = torch.arange(2).repeat_interleave(count // 2)
        marker = 1 - 2 * labels
        inputs[:, 0, 0] = marker
        inputs[:, 3, 0] = -marker
    else:
        quadrants = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        signs = quadrants.repeat_interleave(count // 4, dim=0)
        inputs = signs * (0.35 + 0.65 * torch.rand(count, 2, generator=generator))
        if sample == LINEAR_DATA:
            labels = (inputs[:, 0] > 0).long()
        else:
            labels = ((inputs[:, 0] > 0) != (inputs[:, 1] > 0)).long()
    return TensorDataset(inputs, labels)


class SampleDatasetNode(Node):
    """Create labeled examples with separate, reproducible training and test splits.
    The two point datasets share the same features and differ in their labeling rule.
    The sequence dataset changes class when its two markers trade places.
    """

    title = 'Load Sample Data'
    instructions = 'Choose a dataset, then load the examples.'
    options = (LINEAR_DATA, XOR_DATA, SEQUENCE_DATA)
    init_inputs = [NodeInputType(label='sample', default=Data(LINEAR_DATA))]
    init_outputs = [
        NodeOutputType(label='train_dataset'),
        NodeOutputType(label='test_dataset'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.report = self.instructions

    def update_event(self, inp=-1):
        self.report = self.instructions
        clear_outputs(self)
        if inp >= 0:
            return

        sample = value(self, 0, LINEAR_DATA)
        if sample not in self.options:
            self.report = 'Choose one of the sample datasets.'
            return

        generator = torch.Generator().manual_seed(42)
        train = sample_split(sample, 128, generator)
        test = sample_split(sample, 64, generator)
        shape = ' × '.join(str(size) for size in train.tensors[0].shape[1:])
        self.report = f'{len(train)} train · {len(test)} test\nInput per example: {shape}'
        self.set_output_val(0, Data(train))
        self.set_output_val(1, Data(test))


class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.classifier = nn.Linear(8, 2)

    def forward(self, inputs):
        _, hidden = self.backbone(inputs)
        return self.classifier(hidden[-1])


class SampleModelNode(Node):
    """Build an untrained model for the sample datasets. Build it again to reset
    the weights to their original values before comparing training settings.
    """

    title = 'Build Sample Model'
    instructions = 'Choose a model, then build or reset it.'
    options = (LINEAR_MODEL, MLP_MODEL, RNN_MODEL)
    init_inputs = [NodeInputType(label='architecture', default=Data(LINEAR_MODEL))]
    init_outputs = [NodeOutputType(label='model')]

    def __init__(self, params):
        super().__init__(params)
        self.report = self.instructions

    def update_event(self, inp=-1):
        self.report = self.instructions
        clear_outputs(self)
        if inp >= 0:
            return

        architecture = value(self, 0, LINEAR_MODEL)
        if architecture not in self.options:
            self.report = 'Choose one of the sample models.'
            return

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            if architecture == LINEAR_MODEL:
                model = nn.Linear(2, 2)
                self.report = 'Linear: 2 features → 2 scores'
            elif architecture == MLP_MODEL:
                model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
                self.report = 'Linear: 2 → 8 · ReLU\nLinear: 8 → 2 scores'
            else:
                model = SequenceClassifier()
                self.report = 'RNN: 1 feature per step\nFinal state: 8 → 2 scores'
        self.set_output_val(0, Data(model))
