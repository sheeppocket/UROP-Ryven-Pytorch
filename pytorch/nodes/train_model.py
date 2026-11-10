from ryven.node_env import *
import torch
import torch.nn as nn
from .utils import value, clear_outputs

LOSS_FUNCTIONS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce_logits':    nn.BCEWithLogitsLoss,
    'bce':           nn.BCELoss,
    'mse':           nn.MSELoss,
    'mae':           nn.L1Loss,
    'smooth_l1':     nn.SmoothL1Loss,
}

class TrainModelNode(Node):
    """Learn from labeled training examples. The loss measures how far predictions are
    from the labels. The optimizer uses it to adjust trainable weights. With a fixed
    encoder, only the classifier head learns.
    """
    title = "Train Model"
    init_inputs = [
        NodeInputType(label="model"),
        NodeInputType(label="train_data_loader"),
        NodeInputType(label="loss",          default=Data("cross_entropy")),
        NodeInputType(label="optimizer",     default=Data("Adam")),
        NodeInputType(label="learning_rate", default=Data(0.001)),
        NodeInputType(label="epochs",        default=Data(1)),
    ]
    init_outputs = [
        NodeOutputType(label="trained_model"),
        NodeOutputType(label="loss_history"),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.trained_model = None
        self.revision = 0
        self.training_worker = None

    def remove_event(self):
        self.revision += 1
        if self.training_worker is not None:
            self.training_worker.requestInterruption()

    def training_inputs(self):
        return (
            value(self, 0), value(self, 1),
            str(value(self, 2, 'cross_entropy')).lower(),
            str(value(self, 3, 'Adam')),
            float(value(self, 4, 0.001)), int(value(self, 5, 1)),
        )

    def export_training_code(self):
        from .training_export import training_code

        return training_code(*self.training_inputs())

    def train(self, progress=None):
        """Explicit entry point shared by the GUI and headless callers."""
        self.trained_model = None
        clear_outputs(self)
        model, loader, loss, optimizer, lr, epochs = self.training_inputs()
        trained, history = train_model(model, loader, loss, optimizer, lr, epochs, progress)
        self.trained_model = trained
        self.set_output_val(0, Data(trained))
        self.set_output_val(1, Data(history))
        return history

    def update_event(self, inp=-1):
        # Wiring or editing a graph must never start training implicitly.
        self.revision += 1
        self.trained_model = None
        clear_outputs(self)


def train_model(model, loader, loss_key='cross_entropy', optimizer_name='Adam',
                learning_rate=0.001, epochs=1, progress=None, cancelled=None):
    """Train with labelled batches; return sample-weighted epoch mean losses."""
    if not isinstance(model, nn.Module) or loader is None:
        raise ValueError('Connect a model and a labelled data loader before training')
    if loss_key not in LOSS_FUNCTIONS:
        raise ValueError(f'Unknown loss: {loss_key}')
    if optimizer_name not in ('Adam', 'SGD', 'RMSprop', 'AdamW'):
        raise ValueError(f'Unknown optimizer: {optimizer_name}')
    if learning_rate <= 0 or epochs < 1:
        raise ValueError('Learning rate and epochs must be positive')
    loss_fn = LOSS_FUNCTIONS[loss_key]()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('The model has no trainable parameters')
    optimizer = getattr(torch.optim, optimizer_name)(trainable, lr=learning_rate)
    parameter = next(model.parameters())
    history = []
    model.train()
    for epoch in range(epochs):
        total_loss, count = 0.0, 0
        for batch in loader:
            if cancelled is not None and cancelled():
                raise RuntimeError('Training cancelled')
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError('Training requires (inputs, targets) batches')
            inputs, targets = batch
            inputs = inputs.to(device=parameter.device, dtype=parameter.dtype)
            targets = targets.to(device=parameter.device,
                                 dtype=torch.long if loss_key == 'cross_entropy' else parameter.dtype)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if not torch.isfinite(loss):
                raise ValueError('Training produced a non-finite loss')
            loss.backward()
            optimizer.step()
            size = inputs.shape[0]
            total_loss += loss.item() * size
            count += size
        if count == 0:
            raise ValueError('The data loader is empty')
        history.append(total_loss / count)
        if progress:
            progress(epoch + 1, epochs, history[-1])
    return model, history
