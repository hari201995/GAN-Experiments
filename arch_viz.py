import os

import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _layer_label(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        return f"Linear\n{module.in_features} → {module.out_features}"
    if isinstance(module, nn.Conv2d):
        return (f"Conv2d {module.in_channels}→{module.out_channels}\n"
                f"k={module.kernel_size[0]} s={module.stride[0]}")
    if isinstance(module, nn.LeakyReLU):
        return f"LeakyReLU (α={module.negative_slope})"
    return type(module).__name__


def _is_param_layer(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, nn.Conv2d))


def _collect_layers(model: nn.Module):
    return [
        (name, m)
        for name, m in model.named_modules()
        if not isinstance(m, (nn.Sequential, type(model))) and name
    ]


def _draw_model(ax, layers, title, color):
    n = len(layers)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    for i, (name, module) in enumerate(layers):
        y = n - 1 - i  # top = input, bottom = output

        if _is_param_layer(module):
            rect = mpatches.FancyBboxPatch(
                (0.08, y - 0.34), 0.84, 0.68,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.88,
            )
            ax.add_patch(rect)
            ax.text(0.5, y, _layer_label(module),
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold', linespacing=1.4)
        else:
            ax.text(0.5, y, _layer_label(module),
                    ha='center', va='center', fontsize=7.5,
                    color='#555555', style='italic')

        if i < n - 1:
            ax.annotate(
                '', xy=(0.5, y - 0.62), xytext=(0.5, y - 0.38),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2),
            )


def plot_architecture(generator: nn.Module, discriminator: nn.Module,
                      output_dir: str) -> None:
    g_layers = _collect_layers(generator)
    d_layers = _collect_layers(discriminator)

    n_rows = max(len(g_layers), len(d_layers))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(5, n_rows * 0.75 + 1.5)))
    fig.suptitle('GAN Architecture', fontsize=15, fontweight='bold', y=1.01)

    _draw_model(axes[0], g_layers, 'Generator',      color='#2E6DAD')
    _draw_model(axes[1], d_layers, 'Discriminator',  color='#B5541A')

    plt.tight_layout()
    path = os.path.join(output_dir, 'plots', 'architecture.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved architecture diagram → {path}')
