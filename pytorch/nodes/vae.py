from ryven.node_env import Data, Node, NodeInputType, NodeOutputType
from ryvencore.Base import Event

from .dense_layers import DenseModelSpec, positive_width
from .utils import value


class VariationalLatentNode(Node):
    """Connect the last encoder layer. The exported code learns a mean and log
    variance for each latent dimension and samples a vector for the decoder.
    """

    title = 'Variational Latent'
    instructions = 'Connect the last dense encoder layer.'
    init_inputs = [
        NodeInputType(label='encoder'),
        NodeInputType(label='latent_dimensions', default=Data(8)),
    ]
    init_outputs = [NodeOutputType(label='decoder')]

    def __init__(self, params):
        super().__init__(params)
        self.report_changed = Event(str)

    def rebuilt(self):
        self.update()

    def update_event(self, inp=-1):
        encoder = value(self, 0)
        try:
            if not isinstance(encoder, DenseModelSpec) or not encoder.layers:
                raise ValueError(self.instructions)
            if encoder.encoder or encoder.vae_encoder:
                raise ValueError('Start the encoder with Dense Input.')
            dimensions = positive_width(value(self, 1), 'Latent dimensions')
        except ValueError as error:
            self.set_output_val(0, Data(None))
            self.report_changed.emit(str(error))
            return
        self.set_output_val(0, Data(DenseModelSpec(dimensions, vae_encoder=encoder)))
        self.report_changed.emit(f'{dimensions} latent values per image')


class TrainVAENode(Node):
    """Export Fashion-MNIST VAE training code for Colab or another Python environment."""

    title = 'Train VAE'
    init_inputs = [
        NodeInputType(label='model'),
        NodeInputType(label='train_data_loader'),
        NodeInputType(label='learning_rate', default=Data(0.001)),
        NodeInputType(label='epochs', default=Data(5)),
        NodeInputType(label='kl_weight', default=Data(1.0)),
    ]

    def export_training_code(self):
        from .vae_export import vae_training_code

        return vae_training_code(
            value(self, 0), value(self, 1), float(value(self, 2, 0.001)),
            int(value(self, 3, 5)), float(value(self, 4, 1.0)),
        )
