import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from wgan_model import WGenerator, WCritic, weights_init
from utils import (
    setup_output_dirs, save_image_grid, save_checkpoint,
    plot_loss_curves, log_losses, plot_id_curves, plot_cka,
)
from activation_extractor import extract_activations
from intrinsic_dim import two_nn_id
from cka import linear_cka
from arch_viz import plot_architecture

DATASET_CONFIG = {
    'mnist':   {'image_size': 784,  'img_shape': (1, 28, 28)},
    'cifar10': {'image_size': 3072, 'img_shape': (3, 32, 32)},
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_mnist_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


def get_cifar10_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


def gradient_penalty(critic, real_imgs, fake_imgs, device):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    critic_interp = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train(args):
    device = get_device()
    print(f'Using device: {device}')

    cfg = DATASET_CONFIG[args.dataset]
    image_size = cfg['image_size']
    img_shape  = cfg['img_shape']

    paths = setup_output_dirs(args.output_dir)
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'wgan_training_log.csv')

    if args.dataset == 'mnist':
        dataloader = get_mnist_dataloader(args.data_dir, args.batch_size)
    else:
        dataloader = get_cifar10_dataloader(args.data_dir, args.batch_size)

    generator = WGenerator(latent_dim=args.latent_dim, image_size=image_size).to(device)
    critic = WCritic(image_size=image_size).to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)
    plot_architecture(generator, critic, args.output_dir)

    # WGAN-GP uses Adam (GP removes the need for RMSprop)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    z_id_probe = torch.randn(args.batch_size, args.latent_dim, device=device)
    _probe_batch, _ = next(iter(dataloader))
    id_probe_real = _probe_batch.view(-1, image_size).to(device)

    id_records  = []
    cka_records = []
    global_step = 0

    g_losses_epoch = []
    c_losses_epoch = []

    for epoch in range(1, args.num_epochs + 1):
        current_noise_std = args.noise_std * math.exp(-epoch * args.noise_decay)
        g_losses_batch = []
        c_losses_batch = []

        data_iter = iter(dataloader)
        num_batches = len(dataloader)
        batch_idx = 0

        while batch_idx < num_batches:
            # Train critic n_critic times per generator step
            for _ in range(args.n_critic):
                try:
                    real_batch, _ = next(data_iter)
                except StopIteration:
                    break
                batch_idx += 1

                real_imgs = real_batch.view(-1, image_size).to(device)
                batch_size = real_imgs.size(0)

                c_optimizer.zero_grad()
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake_imgs = generator(z).detach()

                d_input = real_imgs + current_noise_std * torch.randn_like(real_imgs)
                gp = gradient_penalty(critic, real_imgs, fake_imgs, device)
                c_loss = (-torch.mean(critic(d_input))
                          + torch.mean(critic(fake_imgs))
                          + args.lambda_gp * gp)
                c_loss.backward()
                c_optimizer.step()

                c_losses_batch.append(c_loss.item())

            # Train generator
            g_optimizer.zero_grad()
            z = torch.randn(args.batch_size, args.latent_dim, device=device)
            g_loss = -torch.mean(critic(generator(z)))
            g_loss.backward()
            g_optimizer.step()

            g_losses_batch.append(g_loss.item())

            global_step += 1
            if global_step % args.id_log_every == 0:
                generator.eval()
                critic.eval()
                with torch.no_grad():
                    acts = extract_activations(generator, critic,
                                               z_id_probe, id_probe_real)
                generator.train()
                critic.train()
                id_values = {layer: two_nn_id(act) for layer, act in acts.items()}
                id_records.append((global_step, id_values))

                c_layers = ['net.0', 'net.2', 'net.4']
                cka_values = {
                    layer: linear_cka(acts[f'D_fake/{layer}'], acts[f'D_real/{layer}'])
                    for layer in c_layers
                }
                cka_records.append((global_step, cka_values))

        if not g_losses_batch:
            continue

        epoch_g = sum(g_losses_batch) / len(g_losses_batch)
        epoch_c = sum(c_losses_batch) / len(c_losses_batch)
        g_losses_epoch.append(epoch_g)
        c_losses_epoch.append(epoch_c)

        print(f'Epoch [{epoch}/{args.num_epochs}]  G Loss: {epoch_g:.4f}  C Loss: {epoch_c:.4f}')

        with torch.no_grad():
            fake_samples = generator(fixed_noise).cpu()
        save_image_grid(fake_samples, epoch, args.output_dir, img_shape=img_shape)

        log_losses(epoch, epoch_g, epoch_c, log_path)

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                generator, critic, g_optimizer, c_optimizer,
                epoch, {'g_loss': epoch_g, 'c_loss': epoch_c},
                paths['checkpoints'],
            )

    plot_loss_curves(g_losses_epoch, c_losses_epoch, args.output_dir, save=True)
    plot_id_curves(id_records, args.output_dir, save=True)
    plot_cka(cka_records, args.output_dir, save=True)
    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN-GP on MNIST or TFD')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='dataset to train on')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data directory (downloaded automatically if not present)')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_critic', type=int, default=5,
                        help='critic updates per generator update')
    parser.add_argument('--noise_std', type=float, default=0.25,
                        help='initial std of Gaussian noise added to real images before critic')
    parser.add_argument('--noise_decay', type=float, default=0.01,
                        help='exponential decay rate for noise std per epoch')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='gradient penalty coefficient')
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='output directory (defaults to outputs_wgan_<dataset>_annealed)')
    parser.add_argument('--id_log_every', type=int, default=100)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'./outputs_wgan_{args.dataset}_annealed'
    train(args)
