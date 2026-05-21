import torch.nn as nn


class ActivationExtractor:
    """
    Registers forward hooks on every Linear/Conv2d layer in a model and
    accumulates their outputs into a dict: {layer_name: tensor(batch, features)}.

    Usage (one-shot):
        extractor = ActivationExtractor(generator, "G")
        _ = generator(z)
        acts = extractor.activations   # dict populated after forward pass
        extractor.remove_hooks()

    Usage (context manager — hooks are removed automatically):
        with ActivationExtractor(discriminator, "D") as ext:
            _ = discriminator(x)
            acts = ext.activations
    """

    _HOOK_TYPES = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, prefix: str = ""):
        self.prefix = prefix
        self.activations: dict = {}
        self._hooks: list = []

        for name, module in model.named_modules():
            if isinstance(module, self._HOOK_TYPES):
                key = f"{prefix}/{name}" if prefix else name
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(key))
                )

    def _make_hook(self, key: str):
        def hook(module, input, output):
            self.activations[key] = output.detach()
        return hook

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


def extract_activations(
    generator: nn.Module,
    discriminator: nn.Module,
    z,
    real_imgs,
) -> dict:
    """
    Three forward passes returning activations from all three paths:
      G/*       — G layers on fixed noise z
      D_fake/*  — D layers on G(z) (fake images)
      D_real/*  — D layers on real_imgs

    Returns a single merged dict with all keys.
    """
    with ActivationExtractor(generator, "G") as g_ext, \
         ActivationExtractor(discriminator, "D_fake") as d_fake_ext:
        fake_imgs = generator(z)
        discriminator(fake_imgs.detach())
        noise_acts = {**g_ext.activations, **d_fake_ext.activations}

    with ActivationExtractor(discriminator, "D_real") as d_real_ext:
        discriminator(real_imgs)

    return {**noise_acts, **d_real_ext.activations}
