# BRIDGE

BRIDGE is a behavior-guided residual integration model for multimodal recommendation.

## What is included

- `src/`: training code, model code, and evaluation utilities
- `scripts/`: reproducible run scripts for the main datasets
- `data/`: expected dataset format
- `paper/`: optional arXiv source bundle

## Requirements

Use `environment.yaml` for a conda environment, or install from `requirements.txt` if you manage PyTorch and PyG separately.

## Data format

Place each dataset under `data/<dataset_name>/` with:

- `<dataset>.inter`
- `image_feat.npy`
- `text_feat.npy`
- `user_emb.npy`

The interaction file must contain user id, item id, and split label columns.

## Train

From `src/`:

```bash
python main.py --model BRIDGE --dataset baby --gpu_id 0 --seed 2020
```

Use `--dataset sports` or `--dataset elec` for the other benchmarks.

## Notes

The released code keeps the BRIDGE model and its evaluation utilities. Logs, checkpoints, and generated ranking files are ignored by default.

