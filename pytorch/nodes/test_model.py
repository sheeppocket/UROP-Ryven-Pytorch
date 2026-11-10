from ryven.node_env import *
from ryvencore.Base import Event
import torch
from .utils import value, clear_outputs
from .train_model import LOSS_FUNCTIONS


class TestModelNode(Node):
    """Run the model on examples set aside from training. Test accuracy is the fraction
    whose predicted class matches the label. Testing leaves the model weights fixed.
    """

    title = "Check Test Performance"
    tags = ['test model', 'test accuracy']
    instructions = 'Waiting for model and test batches'
    init_inputs = [NodeInputType(label="model"),
                   NodeInputType(label="test_data_loader"),
                   NodeInputType(label="loss_function", default=Data("cross_entropy"))]
    init_outputs = [NodeOutputType(label="test_loss"), NodeOutputType(label="predictions")]

    def __init__(self, params):
        super().__init__(params)
        self.accuracy = None
        self.report_changed = Event(str)

    def update_event(self, inp=-1):
        clear_outputs(self)
        self.accuracy = None
        model, loader, loss_fn = value(self, 0), value(self, 1), value(self, 2, 'cross_entropy')
        if model is None or loader is None:
            self.report_changed.emit(self.instructions)
            return
        was_training = model.training
        try:
            if isinstance(loss_fn, str):
                loss_fn = LOSS_FUNCTIONS[loss_fn]()
            parameter = next(model.parameters())
            total, count, predictions = 0.0, 0, []
            correct, classified = 0, 0
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                        raise ValueError('Testing requires (inputs, targets) batches')
                    inputs, targets = batch
                    inputs = inputs.to(device=parameter.device, dtype=parameter.dtype)
                    targets = targets.to(device=parameter.device,
                        dtype=torch.long if isinstance(loss_fn, torch.nn.CrossEntropyLoss) else parameter.dtype)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    total += loss.item() * inputs.shape[0]
                    count += inputs.shape[0]
                    if isinstance(loss_fn, torch.nn.CrossEntropyLoss) and outputs.ndim == 2 and targets.ndim == 1:
                        valid = targets != loss_fn.ignore_index
                        correct += ((outputs.argmax(dim=1) == targets) & valid).sum().item()
                        classified += valid.sum().item()
                    predictions.append(outputs.cpu())
            if count == 0:
                raise ValueError('The data loader is empty')
            mean_loss = total / count
            report = f'{count} test examples\nMean loss {mean_loss:.3f}'
            if classified:
                self.accuracy = correct / classified
                report += f'\nAccuracy {self.accuracy:.1%} ({correct}/{classified})'
            self.set_output_val(0, Data(mean_loss))
            self.set_output_val(1, Data(predictions))
        except Exception as error:
            print(f'Test Model: {error}')
            clear_outputs(self)
            report = f'Test failed: {error}'
        finally:
            model.train(was_training)
        self.report_changed.emit(report)
