"""
Gaussian Parzen window log-likelihood evaluation for trained GAN generators.

Method (Goodfellow et al. 2014):
  1. Sample N images from G(z), z ~ N(0, I).
  2. Fit a Gaussian KDE with bandwidth sigma over those samples.
  3. Evaluate mean log p(x) on held-out test images under that KDE.
  4. Sigma is chosen by cross-validating on a small validation split of test data.

Usage:
  python parzen_eval.py --checkpoint outputs_wgan_500ep_annealed/checkpoints/checkpoint_epoch_500.pth
  python parzen_eval.py --checkpoint outputs_wgan_500ep_annealed/checkpoints/checkpoint_epoch_500.pth \
      --model wgan --latent_dim 128 --n_samples 10000 --sigma_search 0.1 0.2 0.3 0.5
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets, transforms


# ── Plot style ────────────────────────────────────────────────────────────────

STYLE = dict(
    figure_facecolor='#0f1117',
    axes_facecolor='#1a1d27',
    axes_edgecolor='#2e3347',
    grid_color='#2e3347',
    text_color='#e2e8f0',
    muted_color='#8892a4',
    accent='#6c8fff',
    accent2='#a78bfa',
    success='#34d399',
    warn='#f59e0b',
)

def apply_style(fig, axes):
    fig.patch.set_facecolor(STYLE['figure_facecolor'])
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(STYLE['axes_facecolor'])
        ax.tick_params(colors=STYLE['text_color'], labelsize=9)
        ax.xaxis.label.set_color(STYLE['text_color'])
        ax.yaxis.label.set_color(STYLE['text_color'])
        ax.title.set_color(STYLE['text_color'])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE['axes_edgecolor'])
        ax.grid(True, color=STYLE['grid_color'], linewidth=0.6, linestyle='--', alpha=0.7)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(STYLE['axes_facecolor'])
            legend.get_frame().set_edgecolor(STYLE['axes_edgecolor'])
            for text in legend.get_texts():
                text.set_color(STYLE['text_color'])


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_sigma_search(sigma_candidates, mean_lls, std_lls, best_sigma, plots_dir, run_label):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sigma_candidates, mean_lls,
            color=STYLE['accent'], linewidth=2, marker='o', markersize=7,
            label='Mean log-likelihood (val)')
    ax.fill_between(sigma_candidates,
                    np.array(mean_lls) - np.array(std_lls),
                    np.array(mean_lls) + np.array(std_lls),
                    color=STYLE['accent'], alpha=0.15, label='±1 std')
    ax.axvline(best_sigma, color=STYLE['success'], linewidth=1.8,
               linestyle='--', label=f'Best σ = {best_sigma}')

    ax.set_title(f'Parzen Window — σ Cross-Validation\n{run_label}', fontsize=13, pad=12)
    ax.set_xlabel('Bandwidth σ', fontsize=11)
    ax.set_ylabel('Mean Log-Likelihood (val)', fontsize=11)
    ax.legend(fontsize=9)
    apply_style(fig, ax)

    fig.tight_layout()
    path = os.path.join(plots_dir, 'parzen_sigma_search.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {path}')


def plot_ll_distribution(ll_values, mean_ll, std_ll, sigma, plots_dir, run_label):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(ll_values, bins=40, color=STYLE['accent'], alpha=0.75,
            edgecolor=STYLE['axes_edgecolor'], linewidth=0.5, label='Per-sample log p(x)')
    ax.axvline(mean_ll, color=STYLE['success'], linewidth=2,
               linestyle='-', label=f'Mean = {mean_ll:.2f}')
    ax.axvline(mean_ll - std_ll, color=STYLE['warn'], linewidth=1.4,
               linestyle='--', label=f'±1 std  ({std_ll:.2f})')
    ax.axvline(mean_ll + std_ll, color=STYLE['warn'], linewidth=1.4,
               linestyle='--')

    ax.set_title(f'Parzen Log-Likelihood Distribution  (σ={sigma})\n{run_label}', fontsize=13, pad=12)
    ax.set_xlabel('Log p(x)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=9)
    apply_style(fig, ax)

    fig.tight_layout()
    path = os.path.join(plots_dir, 'parzen_ll_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {path}')


def plot_ll_per_sample(ll_values, mean_ll, sigma, plots_dir, run_label):
    fig, ax = plt.subplots(figsize=(10, 4))

    indices = np.arange(len(ll_values))
    ax.scatter(indices, ll_values, s=4, color=STYLE['accent'], alpha=0.5,
               label='log p(xᵢ)', marker='.')
    ax.axhline(mean_ll, color=STYLE['success'], linewidth=1.8,
               linestyle='-', label=f'Mean = {mean_ll:.2f}')

    ax.set_title(f'Parzen Log-Likelihood per Test Sample  (σ={sigma})\n{run_label}', fontsize=13, pad=12)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Log p(x)', fontsize=11)
    ax.legend(fontsize=9)
    apply_style(fig, ax)

    fig.tight_layout()
    path = os.path.join(plots_dir, 'parzen_ll_per_sample.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {path}')


# ── Parzen window ─────────────────────────────────────────────────────────────

def log_mean_exp(x):
    m = x.max(axis=0)
    return m + np.log(np.exp(x - m).mean(axis=0))


def parzen_log_likelihood(test_samples, gen_samples, sigma):
    D = gen_samples.shape[1]
    log_z = D * (np.log(sigma) + 0.5 * np.log(2 * math.pi))
    ll = []
    for x in test_samples:
        sq_dist = ((gen_samples - x) ** 2).sum(axis=1)
        log_p = -sq_dist / (2 * sigma ** 2) - log_z
        ll.append(log_mean_exp(log_p))
    return np.array(ll)


def cross_validate_sigma(val_samples, gen_samples, sigma_candidates):
    mean_lls, std_lls = [], []
    print(f"  {'sigma':>8}  {'mean log-lik':>14}  {'std':>10}")
    for sigma in sigma_candidates:
        ll = parzen_log_likelihood(val_samples, gen_samples, sigma)
        mean_lls.append(float(ll.mean()))
        std_lls.append(float(ll.std()))
        print(f"  {sigma:>8.4f}  {mean_lls[-1]:>14.4f}  {std_lls[-1]:>10.4f}")
    best_idx = int(np.argmax(mean_lls))
    return sigma_candidates[best_idx], mean_lls, std_lls


# ── Generator loading ──────────────────────────────────────────────────────────

def load_generator(checkpoint_path, model_type, latent_dim, device):
    if model_type == 'wgan':
        from wgan_model import WGenerator
        G = WGenerator(latent_dim=latent_dim).to(device)
    else:
        from model import Generator
        G = Generator(latent_dim=latent_dim).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(state['generator_state_dict'])
    G.eval()
    return G


def sample_generator(G, latent_dim, n_samples, device, batch_size=512):
    samples = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            z = torch.randn(n, latent_dim, device=device)
            samples.append(G(z).cpu().numpy())
    return np.concatenate(samples, axis=0)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return torch.stack([ds[i][0] for i in range(len(ds))]).view(-1, 784).numpy()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Gaussian Parzen window log-likelihood eval')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', choices=['wgan', 'vanilla'], default='wgan')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=200)
    parser.add_argument('--sigma_search', type=float, nargs='+',
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    # Derive plots dir from checkpoint path: <output_dir>/plots/
    output_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    plots_dir  = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Label for plot titles
    run_label = os.path.basename(output_dir)

    print(f'Device     : {device}')
    print(f'Checkpoint : {args.checkpoint}')
    print(f'Plots dir  : {plots_dir}')

    print(f'\nLoading generator…')
    G = load_generator(args.checkpoint, args.model, args.latent_dim, device)

    print(f'Sampling {args.n_samples} images from G…')
    gen_samples = sample_generator(G, args.latent_dim, args.n_samples, device)

    print(f'\nLoading MNIST test set…')
    test_data = load_test_data(args.data_dir)
    np.random.shuffle(test_data)
    val_data  = test_data[:args.n_val]
    eval_data = test_data[args.n_val:args.n_val + args.n_test]
    print(f'Val : {len(val_data)}  |  Eval : {len(eval_data)}')

    if args.sigma is None:
        print(f'\nCross-validating sigma…')
        sigma, mean_lls, std_lls = cross_validate_sigma(val_data, gen_samples, args.sigma_search)
        print(f'Best sigma : {sigma}')
        print(f'\nSaving plots…')
        plot_sigma_search(args.sigma_search, mean_lls, std_lls, sigma, plots_dir, run_label)
    else:
        sigma = args.sigma
        print(f'Using fixed sigma : {sigma}')

    print(f'\nEvaluating on {len(eval_data)} test samples…')
    ll_values = parzen_log_likelihood(eval_data, gen_samples, sigma)
    mean_ll   = float(ll_values.mean())
    std_ll    = float(ll_values.std())
    stderr    = std_ll / math.sqrt(len(eval_data))

    print(f'\nSaving plots…')
    plot_ll_distribution(ll_values, mean_ll, std_ll, sigma, plots_dir, run_label)
    plot_ll_per_sample(ll_values, mean_ll, sigma, plots_dir, run_label)

    print(f'\n{"─"*42}')
    print(f'  Parzen Log-Likelihood  ({run_label})')
    print(f'  sigma  : {sigma}')
    print(f'  mean   : {mean_ll:.4f}')
    print(f'  std    : {std_ll:.4f}')
    print(f'  stderr : {stderr:.4f}')
    print(f'{"─"*42}')


if __name__ == '__main__':
    main()
