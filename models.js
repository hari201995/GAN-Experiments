/**
 * MODELS config — edit here to add new runs.
 * Both index.html and dashboard.html read from this file.
 *
 * Fields:
 *   name  — display name shown in cards and tabs
 *   tag   — short label (GAN type, variant, etc.)
 *   path  — folder that contains outputs/plots/ and outputs/images/
 *   desc  — one-line description shown on the index card
 */
const MODELS = [
  {
    name: 'Vanilla GAN',
    tag:  'no label smoothing',
    path: 'wo_label_smoothing/outputs',
    desc: 'Baseline GAN trained on MNIST with BCELoss and Adam.',
  },
  // ── Add new runs below ──────────────────────────────────────────
  {
    name: 'WGAN-GP — Clean',
    tag:  'wasserstein · clean',
    path: 'outputs_wgan_clean',
    desc: 'WGAN-GP on MNIST, 375 epochs, no noise on real images (noise_std=0). Adam (β=0.0/0.9), 5 critic steps per G step, λ_gp=10. Clean baseline.',
  },
  {
    name: 'WGAN — Noisy Real Input',
    tag:  'ablation · noisy-real',
    path: 'outputs_wgan_375ep_clean',
    desc: '[Ablation] WGAN-GP critic receives real images with 0.25×N(0,1) Gaussian noise. Gradient penalty uses clean real images. Adam (β=0.0/0.9), 5 critic steps per G step, λ_gp=10, 375 epochs.',
  },
  {
    name: 'WGAN-GP — Annealed Noise',
    tag:  '500ep · annealed',
    path: 'outputs_wgan_500ep_annealed',
    desc: 'WGAN-GP with annealed Gaussian noise on real images: σ = 0.25·e^(−epoch·0.01). Noise weakens D early then decays to ~0 by epoch 500. GP uses clean real images. Adam (β=0.0/0.9), 5 critic steps, λ_gp=10.',
  },
];
