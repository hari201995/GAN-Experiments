import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Generator, Discriminator, weights_init
from utils import (
    setup_output_dirs, save_image_grid, save_checkpoint,
    plot_loss_curves, log_losses, plot_id_curves, plot_cka,
)
from activation_extractor import extract_activations
from intrinsic_dim import two_nn_id
from cka import linear_cka
from arch_viz import plot_architecture


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


def train(args):
    device = get_device()
    print(f'Using device: {device}')

    paths = setup_output_dirs(args.output_dir)
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'training_log.csv')

    dataloader = get_dataloader(args.data_dir, args.batch_size)

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    plot_architecture(generator, discriminator, args.output_dir)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=args.betas)

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    # fixed probes used exclusively for ID estimation — never touch optimizer
    z_id_probe = torch.randn(args.batch_size, args.latent_dim, device=device)
    _probe_batch, _ = next(iter(dataloader))
    id_probe_real = _probe_batch.view(-1, 784).to(device)

    id_records  = []   # list of (global_step, {layer: id_value})
    cka_records = []   # list of (global_step, {d_layer: cka_value})
    global_step = 0

    g_losses_epoch = []
    d_losses_epoch = []

    for epoch in range(1, args.num_epochs + 1):
        g_losses_batch = []
        d_losses_batch = []

        for real_batch, _ in tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs}', leave=False):
            real_imgs = real_batch.view(-1, 784).to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            d_optimizer.zero_grad()
            d_input = real_imgs + 0.25 * torch.randn_like(real_imgs)
            d_loss_real = criterion(discriminator(d_input), real_labels)
            z = torch.randn(batch_size, args.latent_dim, device=device)
            d_loss_fake = criterion(discriminator(generator(z).detach()), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            g_loss = criterion(discriminator(generator(z)), torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            g_optimizer.step()

            g_losses_batch.append(g_loss.item())
            d_losses_batch.append(d_loss.item())

            global_step += 1
            if global_step % args.id_log_every == 0:
                generator.eval()
                discriminator.eval()
                with torch.no_grad():
                    acts = extract_activations(generator, discriminator,
                                               z_id_probe, id_probe_real)
                generator.train()
                discriminator.train()
                id_values = {layer: two_nn_id(act) for layer, act in acts.items()}
                id_records.append((global_step, id_values))

                d_layers = ['net.0', 'net.2', 'net.4']
                cka_values = {
                    layer: linear_cka(acts[f'D_fake/{layer}'], acts[f'D_real/{layer}'])
                    for layer in d_layers
                }
                cka_records.append((global_step, cka_values))

        epoch_g = sum(g_losses_batch) / len(g_losses_batch)
        epoch_d = sum(d_losses_batch) / len(d_losses_batch)
        g_losses_epoch.append(epoch_g)
        d_losses_epoch.append(epoch_d)

        print(f'Epoch [{epoch}/{args.num_epochs}]  G Loss: {epoch_g:.4f}  D Loss: {epoch_d:.4f}')

        with torch.no_grad():
            fake_samples = generator(fixed_noise).cpu()
        save_image_grid(fake_samples, epoch, args.output_dir)

        log_losses(epoch, epoch_g, epoch_d, log_path)

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                generator, discriminator, g_optimizer, d_optimizer,
                epoch, {'g_loss': epoch_g, 'd_loss': epoch_d},
                paths['checkpoints'],
            )

    plot_loss_curves(g_losses_epoch, d_losses_epoch, args.output_dir, save=True)
    plot_id_curves(id_records, args.output_dir, save=True)
    plot_cka(cka_records, args.output_dir, save=True)
    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vanilla GAN on MNIST')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--id_log_every', type=int, default=100)
    args = parser.parse_args()
    args.betas = (0.5, 0.999)
    train(args)
