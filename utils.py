import os
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
from tqdm.notebook import tqdm
from IPython.display import clear_output


def setup_output_dirs(base_dir):
    paths = {
        'images': os.path.join(base_dir, 'images'),
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'plots': os.path.join(base_dir, 'plots'),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_image_grid(fake_images, epoch, output_dir, nrow=8, img_shape=(1, 28, 28)):
    imgs = fake_images.view(-1, *img_shape)
    imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
    grid = make_grid(imgs, nrow=nrow, normalize=False, padding=2)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
    img = Image.fromarray(grid_np)
    img.save(os.path.join(output_dir, 'images', f'epoch_{epoch:03d}.png'))
    return grid


def plot_loss_curves(g_losses, d_losses, output_dir, save=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g_losses, label='Generator Loss', color='steelblue')
    ax.plot(d_losses, label='Discriminator Loss', color='darkorange')
    ax.set_title('GAN Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save:
        fig.savefig(os.path.join(output_dir, 'plots', 'loss_curves.png'), dpi=150, bbox_inches='tight')
    return fig


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                    epoch, loss_dict, checkpoint_dir):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        **loss_dict,
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(state, path)


def load_checkpoint(checkpoint_path, generator, discriminator,
                    g_optimizer, d_optimizer, device):
    state = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state['generator_state_dict'])
    discriminator.load_state_dict(state['discriminator_state_dict'])
    g_optimizer.load_state_dict(state['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(state['d_optimizer_state_dict'])
    return state['epoch'], state['g_loss'], state['d_loss']


def _ema(values, alpha=0.1):
    smoothed, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


def plot_id_curves(id_records, output_dir, ema_alpha=0.1, save=True):
    """
    id_records: list of (global_step, {layer_name: float})
    Plots one subplot per group (G, D_fake, D_real), one EMA line per layer.
    """
    if not id_records:
        return

    steps = [r[0] for r in id_records]
    layer_names = list(id_records[0][1].keys())

    groups = {'G': [], 'D_fake': [], 'D_real': []}
    for name in layer_names:
        prefix = name.split('/')[0]
        if prefix in groups:
            groups[prefix].append(name)

    group_colors = {
        'G':      ['#1f77b4', '#4a9fd4', '#7ec8e3', '#aee3f5'],
        'D_fake': ['#d62728', '#e8604c', '#f4977a', '#f9c4ae'],
        'D_real': ['#2ca02c', '#5bbf5b', '#8fd68f', '#bceabc'],
    }
    group_titles = {
        'G':      'Generator — Fixed Noise Probe',
        'D_fake': 'Discriminator — Fake Images (G output)',
        'D_real': 'Discriminator — Real Images Probe',
    }

    active_groups = [g for g in ('G', 'D_fake', 'D_real') if groups[g]]
    fig, axes = plt.subplots(len(active_groups), 1,
                             figsize=(12, 4 * len(active_groups)),
                             sharex=True)
    if len(active_groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, active_groups):
        for i, layer in enumerate(groups[group]):
            values = [r[1].get(layer, float('nan')) for r in id_records]
            smoothed = _ema(values, alpha=ema_alpha)
            color = group_colors[group][i % len(group_colors[group])]
            label = layer.split('/')[1]
            ax.plot(steps, smoothed, label=label, color=color, linewidth=1.8)

        ax.set_title(group_titles[group], fontsize=11, fontweight='bold')
        ax.set_ylabel('Intrinsic Dimensionality')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Training Step')
    fig.suptitle('Intrinsic Dimensionality over Training (EMA)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'plots', 'id_curves.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved ID curves → {path}')
    plt.close(fig)


def plot_cka(cka_records, output_dir, ema_alpha=0.1, save=True):
    """
    cka_records: list of (global_step, {layer_name: float})
    Line plot of EMA-smoothed CKA(D_fake, D_real) per D layer over training.
    """
    if not cka_records:
        return

    steps = [r[0] for r in cka_records]
    layer_names = list(cka_records[0][1].keys())

    colors = ['#e6194b', '#f58231', '#3cb44b', '#4363d8', '#911eb4']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, layer in enumerate(layer_names):
        values = [r[1].get(layer, float('nan')) for r in cka_records]
        smoothed = _ema(values, alpha=ema_alpha)
        ax.plot(steps, smoothed, label=layer, color=colors[i % len(colors)], linewidth=1.8)

    ax.set_title('CKA: D(fake) vs D(real) per Layer (EMA)', fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('CKA  (1 = fake ≈ real,  0 = clearly different)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'plots', 'cka.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved CKA plot → {path}')
    plt.close(fig)


def wasserstein_critic_loss(real_scores, fake_scores):
    """Critic loss: minimize mean(fake) - mean(real)  ↔  maximize separation."""
    return fake_scores.mean() - real_scores.mean()


def wasserstein_generator_loss(fake_scores):
    """Generator loss: minimize -mean(fake)  ↔  fool the critic."""
    return -fake_scores.mean()


def clip_critic_weights(critic, clip_val=0.01):
    for p in critic.parameters():
        p.data.clamp_(-clip_val, clip_val)


def train_wgan(generator, critic, dataloader, config, device, paths):
    """
    WGAN training loop.
    config keys used: latent_dim, num_epochs, lr, checkpoint_every,
                      output_dir, log_path, n_critic, clip_val.
    """
    n_critic  = config.get('n_critic', 5)
    clip_val  = config.get('clip_val', 0.01)
    lr        = config.get('lr', 0.00005)

    # WGAN paper uses RMSprop, not Adam
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)
    c_optimizer = torch.optim.RMSprop(critic.parameters(),    lr=lr)

    fixed_noise = torch.randn(64, config['latent_dim'], device=device)
    g_losses_epoch, c_losses_epoch = [], []

    for epoch in range(1, config['num_epochs'] + 1):
        g_losses_batch, c_losses_batch = [], []
        gen_iter = iter(dataloader)

        for real_batch, _ in tqdm(dataloader, desc=f'Epoch {epoch}/{config["num_epochs"]}', leave=False):
            real_imgs = real_batch.view(-1, 784).to(device)
            batch_size = real_imgs.size(0)

            # --- Critic update (n_critic steps) ---
            for _ in range(n_critic):
                c_optimizer.zero_grad()
                z = torch.randn(batch_size, config['latent_dim'], device=device)
                fake_imgs = generator(z).detach()
                c_loss = wasserstein_critic_loss(critic(real_imgs), critic(fake_imgs))
                c_loss.backward()
                c_optimizer.step()
                clip_critic_weights(critic, clip_val)

            c_losses_batch.append(c_loss.item())

            # --- Generator update ---
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            g_loss = wasserstein_generator_loss(critic(generator(z)))
            g_loss.backward()
            g_optimizer.step()
            g_losses_batch.append(g_loss.item())

        epoch_g = sum(g_losses_batch) / len(g_losses_batch)
        epoch_c = sum(c_losses_batch) / len(c_losses_batch)
        g_losses_epoch.append(epoch_g)
        c_losses_epoch.append(epoch_c)

        with torch.no_grad():
            fake_samples = generator(fixed_noise).cpu()
        grid = save_image_grid(fake_samples, epoch, config['output_dir'])
        log_losses(epoch, epoch_g, epoch_c, config['log_path'])

        if epoch % config['checkpoint_every'] == 0:
            save_checkpoint(
                generator, critic, g_optimizer, c_optimizer,
                epoch, {'g_loss': epoch_g, 'd_loss': epoch_c},
                paths['checkpoints'],
            )

        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        grid_np = grid.permute(1, 2, 0).numpy()
        axes[0].imshow(grid_np, cmap='gray')
        axes[0].set_title(f'WGAN Generated Samples — Epoch {epoch}')
        axes[0].axis('off')
        axes[1].plot(g_losses_epoch, label='G Loss', color='steelblue')
        axes[1].plot(c_losses_epoch, label='Critic Loss', color='darkorange')
        axes[1].set_title('WGAN Loss Curves (Wasserstein distance)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f'Epoch [{epoch}/{config["num_epochs"]}]  G Loss: {epoch_g:.4f}  Critic Loss: {epoch_c:.4f}')

    plot_loss_curves(g_losses_epoch, c_losses_epoch, config['output_dir'], save=True)
    print('WGAN training complete.')
    return g_losses_epoch, c_losses_epoch


def log_losses(epoch, g_loss, d_loss, log_path):
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch', 'g_loss', 'd_loss'])
        writer.writerow([epoch, f'{g_loss:.6f}', f'{d_loss:.6f}'])
