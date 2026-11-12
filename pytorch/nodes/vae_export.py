from math import isfinite
from pprint import pformat
from textwrap import dedent

from .dense_layers import DenseModelSpec
from .torch_utils_data import RemoteBatches, RemoteFashionMNIST
from .training_export import dense_module_lines


def vae_training_code(model, loader, learning_rate, epochs, kl_weight):
    if not isinstance(model, DenseModelSpec) or model.vae_encoder is None or not model.layers:
        raise ValueError('Connect the last decoder layer from a Variational Latent node.')
    encoder = model.vae_encoder
    if encoder.input_features != 784 or model.layers[-1] != (784, 'None'):
        raise ValueError('Use 784 input features and a final decoder layer of 784 neurons with no activation.')
    if not isinstance(loader, RemoteBatches) or not isinstance(loader.dataset, RemoteFashionMNIST):
        raise ValueError('Connect Fashion-MNIST through Group Examples into Batches.')
    if not isfinite(learning_rate) or learning_rate <= 0 or epochs < 1:
        raise ValueError('Learning rate and epochs must be positive.')
    if not isfinite(kl_weight) or kl_weight < 0:
        raise ValueError('KL weight must be a finite non-negative number.')

    settings = {
        'data_root': loader.dataset.root,
        'batch_size': loader.batch_size,
        'shuffle': loader.shuffle,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'kl_weight': kl_weight,
        'seed': 42,
        'checkpoint': 'fashion_mnist_vae.pth',
        'preview': 'fashion_mnist_vae.png',
    }
    config = f'LATENT_DIM = {model.input_features}\nSETTINGS = {pformat(settings, sort_dicts=False)}'
    encoder_width = encoder.layers[-1][0]
    model_code = '\n'.join((
        'class VAE(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
        '        self.encoder = nn.Sequential(',
        *dense_module_lines(encoder, '            '),
        '        )',
        f'        self.mean = nn.Linear({encoder_width}, LATENT_DIM)',
        f'        self.log_variance = nn.Linear({encoder_width}, LATENT_DIM)',
        '        self.decoder = nn.Sequential(',
        *dense_module_lines(model, '            '),
        '        )',
    ))
    return '\n\n'.join((IMPORTS, config, model_code + '\n\n' + MODEL_METHODS, TRAINING_SCRIPT))


IMPORTS = dedent('''\
    # Paste into a Colab cell or run as a Python script. Requires PyTorch and torchvision.
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
''')


MODEL_METHODS = '''\
    def encode(self, images):
        features = self.encoder(images.flatten(1))
        return self.mean(features), self.log_variance(features)

    def decode(self, latent):
        return self.decoder(latent).reshape(-1, 1, 28, 28)

    def forward(self, images):
        mean, log_variance = self.encode(images)
        std = torch.exp(0.5 * log_variance)
        latent = mean + std * torch.randn_like(std)
        return self.decode(latent), mean, log_variance
'''


TRAINING_SCRIPT = dedent('''\
    class FashionMNIST(datasets.FashionMNIST):
        # Prefer the dataset publisher's HTTPS mirror.
        mirrors = [
            'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
            *datasets.FashionMNIST.mirrors,
        ]


    def run_epoch(model, batches, device, optimizer=None):
        training = optimizer is not None
        model.train(training)
        total_reconstruction, total_kl, examples = 0.0, 0.0, 0
        with torch.set_grad_enabled(training):
            for images, _ in batches:
                images = images.to(device)
                logits, mean, log_variance = model(images)
                reconstruction = F.binary_cross_entropy_with_logits(logits, images, reduction='sum')
                kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum()
                if training:
                    loss = (reconstruction + SETTINGS['kl_weight'] * kl) / images.shape[0]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_reconstruction += reconstruction.item()
                total_kl += kl.item()
                examples += images.shape[0]
        return {'reconstruction': total_reconstruction / examples, 'kl': total_kl / examples}


    @torch.inference_mode()
    def save_preview(model, test_batches, device):
        images, _ = next(iter(test_batches))
        images = images[:8].to(device)
        mean, _ = model.encode(images)
        reconstructions = model.decode(mean).sigmoid()
        latent = torch.randn(images.shape[0], LATENT_DIM, device=device)
        generated = model.decode(latent).sigmoid()
        grid = torch.cat((images, reconstructions, generated)).cpu()
        save_image(grid, SETTINGS['preview'], nrow=len(images))
        print(f"Saved {SETTINGS['preview']} (originals, reconstructions, generated samples)", flush=True)


    def main():
        torch.manual_seed(SETTINGS['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pixels = transforms.ToTensor()
        train = FashionMNIST(SETTINGS['data_root'], train=True, transform=pixels, download=True)
        test = FashionMNIST(SETTINGS['data_root'], train=False, transform=pixels, download=True)
        train_batches = DataLoader(train, batch_size=SETTINGS['batch_size'], shuffle=SETTINGS['shuffle'])
        test_batches = DataLoader(test, batch_size=SETTINGS['batch_size'])
        model = VAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'])
        print(f'Training on {device} with {len(train)} images', flush=True)
        history = []
        for epoch in range(1, SETTINGS['epochs'] + 1):
            scores = run_epoch(model, train_batches, device, optimizer)
            history.append(scores)
            print(f"Epoch {epoch}/{SETTINGS['epochs']} | reconstruction={scores['reconstruction']:.2f}"
                  f" | KL={scores['kl']:.2f}", flush=True)
        test_scores = run_epoch(model, test_batches, device)
        print(f"Test ({len(test)} images) | reconstruction={test_scores['reconstruction']:.2f}"
              f" | KL={test_scores['kl']:.2f}", flush=True)
        save_preview(model, test_batches, device)
        torch.save({
            'model_state_dict': model.cpu().state_dict(),
            'history': history,
            'test': test_scores,
            'latent_dimensions': LATENT_DIM,
        }, SETTINGS['checkpoint'])
        print(f"Saved {SETTINGS['checkpoint']}", flush=True)


    if __name__ == '__main__':
        main()
''')
