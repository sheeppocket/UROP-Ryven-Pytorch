import torch
from torch.utils.data import TensorDataset

from ryven.node_env import Data, Node, NodeInputType, NodeOutputType

from .utils import clear_outputs, value


LINEAR_DATA = 'Linear boundary'
XOR_DATA = 'XOR quadrants'


def sample_split(sample, count, generator):
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
    """

    title = 'Load Sample Data'
    instructions = 'Choose a dataset, then load the examples.'
    options = (LINEAR_DATA, XOR_DATA)
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
        train._ryven_sample = sample
        test = sample_split(sample, 64, generator)
        shape = ' × '.join(str(size) for size in train.tensors[0].shape[1:])
        self.report = f'{len(train)} train · {len(test)} test\nInput per example: {shape}'
        self.set_output_val(0, Data(train))
        self.set_output_val(1, Data(test))
