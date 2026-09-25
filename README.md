# PC-CLOT: Point-guided Closed-Loop Optimal Transport

Official PyTorch implementation of **PC-CLOT** — a point-supervised polyp segmentation framework for colonoscopy images. Given an RGB image and a few foreground/background click points, PC-CLOT produces a dense pixel-level polyp mask through optimal transport and closed-loop prototype refinement, without requiring full manual annotations.

## Overview

Full pixel-level annotations are expensive in medical imaging. PC-CLOT propagates sparse point clicks into dense pseudo-labels via **Point-guided Partial Optimal Transport (PPOT)**, then refines them through a two-stage **Transport-Induced Prototype Refinement (TIPR)** closed-loop architecture.

### Pipeline

1. **FuseUNet Encoder** extracts multi-scale features from the input image + point maps.
2. **PCSC Skip Fusion** fuses skip connections into a shared memory grid via predictor-corrector ODE blocks.
3. **PPOT** solves a Sinkhorn-regularized optimal transport problem with learned geodesic costs to produce soft pseudo-masks from sparse points.
4. **Prototype Pooling** aggregates foreground, background, boundary, and uncertainty prototypes from the transport plan.
5. **Cross-Attention Refinement** refines features via cross-attention with the learned prototypes.

## Installation

```bash
# PyTorch (CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install numpy pillow tqdm

# Optional: experiment tracking
pip install swanlab
```

No `pydijkstra` build is required — a pure-Python fallback is included.

## Datasets

Three polyp segmentation datasets are supported:

| Dataset | `--dataset` flag |
|---------|-----------------|
| CVC-ClinicDB | `cvc` |
| Kvasir-SEG | `kvasir` |
| CVC-ColonDB | `cvc_clon` |

Expected directory structure:

```
data/
  cvc/
    images/        # RGB colonoscopy frames
    masks/         # GT masks (for point sampling + evaluation only)
  kvasir/
    images/
    masks/
  cvc_clon/
    images/
    masks/
```

**Note:** GT masks are only used for point-click sampling and evaluation metrics. No mask supervision is applied during training — this is a true point-supervised setting.

## Training

```bash
python main.py \
  --mode train \
  --dataset cvc \
  --data_root ./data \
  --split_dir ./splits
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--num_fg_points` | 1 | Foreground click point per sample |
| `--num_bg_points` | 1 | Background click point per sample |
| `--use_swanlab` | — | Enable SwanLab experiment tracking |

Checkpoints are saved to `<save_dir>/checkpoints/latest.pth`. Evaluation runs every 10 epochs starting after epoch 40.

## Testing

```bash
python main.py \
  --mode test \
  --dataset cvc \
  --data_root ./data \
  --split_dir ./splits \
  --checkpoint <path_to_checkpoint>
```

Outputs per-sample prediction masks, visualizations, and a `metrics.json` with Dice, IoU, precision, recall, specificity, F1, and MAE.

Test a single sample or the full dataset:

```bash
python main.py --mode test --dataset cvc ... --test_sample_name image001.png
python main.py --mode test --dataset cvc ... --test_full_dataset
```

The V4 backbone input is a five-channel tensor formed by concatenating the RGB
image with separate foreground-click and background-click maps. Final training
and test evaluation run the same checkpoint with five click-sampling seeds by
default (`seed` through `seed+4`) and write per-seed metrics plus their mean and
population variance to `test_predictions/multi_seed/multi_seed_metrics.json`.
Use `--eval_seeds 11 22 33` to choose a different seed set.

## Ablation Experiments

Controlled ablation scripts are in `ablations/`:

| Directory | What it studies |
|-----------|----------------|
| `ppot_cost_components/` | OT cost variants (spatial, semantic, boundary, feature) |
| `closed_loop/` | Refinement modes (one-pass, no-prototype, TIPR) |
| `prototype_composition/` | Prototype sets (fg+bg, +boundary, +uncertainty) |
| `sinkhorn_iterations/` | Sinkhorn iteration counts |
| `geodesic_cost/` | Euclidean vs. Dijkstra spatial distance |

Run from the matching directory or use `table2_variants.py` for the unified experiment runner.

### CLI Ablation Flags

| Flag | Options |
|------|---------|
| `--closed_loop_ablation` | `one_pass_ppot`, `closed_loop_no_proto`, `closed_loop_tipr` |
| `--prototype_ablation` | `fg_bg`, `fg_bg_boundary`, `fg_bg_uncertainty`, `fg_bg_boundary_uncertainty` |
| `--ppot_cost_ablation` | `spatial_only`, `spatial_semantic`, `spatial_semantic_boundary`, `spatial_semantic_boundary_feature` |
| `--disable_pcsc` | Ablate PCSC to static skip fusion |
| `--disable_ppot` | Point-only baseline without OT pseudo-masks |

## Project Structure

```
PC-CLOT/
├── main.py                  # Entry point (train/test)
├── config.py                # Base configuration
├── configs/                 # Per-model config overrides
├── models/
│   ├── point2mask_v4.py     # Main V4 model (two-stage closed-loop)
│   ├── point2mask_ot.py     # PPOT optimal transport module
│   ├── prototype_pooling.py # Prototype aggregation
│   ├── clot_refinement.py   # Cross-attention refinement
│   └── modules/             # Building blocks (UNet, heads, attention)
├── losses/
│   └── point2mask_v4_loss.py
├── data/
│   └── dataset.py           # Data loading and point sampling
├── train/
│   └── trainer.py           # Training and evaluation loops
├── utils.py                 # Metrics, visualization, optimizers
└── ablations/               # Controlled experiment scripts and presets
```

## License

This project is released for research purposes.
