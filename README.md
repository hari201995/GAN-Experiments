# GAN Experiments — MNIST

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Exploring Generative Adversarial Networks on MNIST: from a vanilla GAN baseline to WGAN-GP with ablations on noise injection and annealing strategies.

**Live dashboard → [hari201995.github.io/GAN-Experiments](https://hari201995.github.io/GAN-Experiments/)**

---

## Experiments

| Model | Tag | Description |
|---|---|---|
| Vanilla GAN | no label smoothing | Baseline GAN with BCELoss and Adam. |
| WGAN-GP — Clean | wasserstein · clean | WGAN-GP, 375 epochs, no noise on real images. Clean baseline. |
| WGAN — Noisy Real Input | ablation · noisy-real | Critic receives real images with 0.25×N(0,1) Gaussian noise. GP uses clean real images. 375 epochs. |
| WGAN-GP — Annealed Noise | 500ep · annealed | Noise on real images annealed as σ = 0.25·e^(−epoch·0.01), decaying to ~0 by epoch 500. |

---

## Analysis

Each experiment includes:

- **Loss curves** — Generator vs Discriminator/Critic training loss
- **Intrinsic dimensionality** — ID curves across training
- **CKA** — Centered Kernel Alignment across discriminator layers
- **Parzen window log-likelihood** — σ cross-validation, distribution, and per-sample scores
- **Architecture diagram** — Generator and discriminator structure
- **Epoch image viewer** — Scrub or play through generated images across all epochs

---

## Setup

**Requirements**

```bash
pip install -r requirements.txt
```

**Train Vanilla GAN**

```bash
python train.py
```

**Train WGAN-GP**

```bash
python wgan_train.py
```

**Run analysis** (CKA, intrinsic dim, Parzen eval)

```bash
python cka.py
python intrinsic_dim.py
python parzen_eval.py
```

---

## Dashboard

The dashboard is a static HTML/JS site — no server required.

**View locally** — just open `index.html` in your browser.

**Add a new model run:**

1. Add an entry to `models.js`
2. Run `python generate_manifest.py` to rebuild the file index
3. Commit and push

---

## Stack

- PyTorch
- MNIST dataset
- Vanilla HTML/CSS/JS (no frameworks)

## AI
- Claude Sonnet for dashboarding and framework

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt this material for any purpose, provided you give appropriate credit to the original author.
