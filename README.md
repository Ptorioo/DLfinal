# DL-final-branch-c

Branch C: Patch-Level Forensic Branch for AI-generated image detection.

This project uses a two-branch fusion detector:

- Branch A: semantic/global image features
- Branch C: patch-level forensic features selected by FFT energy

## Dataset Layout

Expected local layout:

```text
dataset/
  cifake/
    train/
      REAL/
      FAKE/
    test/
      REAL/
      FAKE/
  tiny-genimage/
    imagenet_ai_0419_biggan/
      train/{nature,ai}/
      val/{nature,ai}/
    ...
```

Labels:

- `REAL` / `nature` = `0`
- `FAKE` / `ai` = `1`

## Training

Training entry point:

```bash
python train.py --help
```

Important options:

- `--dataset`: choose one or more datasets. Supported names are `cifake` and `tiny-genimage`.
- `--dataset-root`: root folder that contains `dataset/cifake/` and `dataset/tiny-genimage/`
- `--generators`: only for `tiny-genimage`, to select a subset of generators such as `sdv5`
- `--augment` / `--no-augment`: enable or disable training data augmentation
- `--output-dir`: where checkpoints and logs are written
- `--val-fraction`: fraction of the training split used as validation

Current behavior:

- Training can use one dataset or merge CIFAKE and Tiny-GenImage into one combined training set.
- To train on one dataset, pass one name: `--dataset cifake` or `--dataset tiny-genimage`.
- To train on both datasets, pass both names: `--dataset cifake tiny-genimage`.
- You can also use `--dataset both` as a shortcut for combining both datasets.
- For Tiny-GenImage, `--generators` filters which generator folders are included.
- Validation is split from the selected training records according to `--val-fraction`.
- Augmentation affects only the training split. Validation and evaluation use deterministic preprocessing.

Augmentation:

- Without `--augment`, images are converted to RGB, resized, converted to tensors, and normalized.
- With `--augment`, training images additionally use random horizontal flip with probability `0.5`.
- With `--augment`, training images additionally use Gaussian blur with probability `0.25`, `kernel_size=3`, and `sigma=(0.1, 0.8)`.
- Use `--no-augment` to explicitly disable augmentation when a config file sets `"augment": true`.

Outputs are written to `--output-dir`, defaulting to `runs/fusion_a_c`.
Each run saves:

- `best.pt`
- `last.pt`
- `history.json`
- `config.resolved.json`

Examples:

```bash
python train.py --config configs/cifake_branch_c.json
```

```bash
python train.py --dataset cifake --epochs 10 --batch-size 128 --patch-size 16 --stride 8 --top-k 4 --output-dir runs/cifake_branch_c
```

```bash
python train.py --dataset cifake --augment --epochs 10 --batch-size 128 --patch-size 16 --stride 8 --top-k 4 --output-dir runs/cifake_aug_branch_c
```

```bash
python train.py --dataset tiny-genimage --generators sdv5 --augment --epochs 10 --batch-size 64 --forensic-size 224 --patch-size 32 --stride 16 --top-k 8 --output-dir runs/tiny_sdv5_aug_branch_c
```

```bash
python train.py --dataset cifake tiny-genimage --augment --epochs 10 --batch-size 64 --forensic-size 224 --patch-size 32 --stride 16 --top-k 8 --output-dir runs/cifake_tiny_aug_fusion_a_c
```

## Evaluate

Evaluation entry point:

```bash
python evaluate.py --help
```

Important options:

- `--checkpoint`: required, points to a `.pt` file such as `runs/.../best.pt`
- `--dataset`: choose `cifake` or `tiny-genimage`
- `--split`: choose `train`, `val`, or `test`
- `--generators`: optional filter for Tiny-GenImage

Current behavior:

- Evaluate loads the checkpoint and prints metrics to the terminal.
- It does not automatically write an evaluation file.
- If you want to save the output, redirect stdout yourself.

Examples:

```bash
python evaluate.py --checkpoint runs/cifake_fusion_a_c/best.pt --dataset cifake --split test
```

```bash
python evaluate.py --checkpoint runs/cifake_fusion_a_c/best.pt --dataset tiny-genimage --split val --forensic-size 224
```

Cross-domain example with generator filtering:

```bash
python evaluate.py --checkpoint runs/cifake_fusion_a_c/best.pt --dataset tiny-genimage --split val --generators sdv5 --forensic-size 224
```

```bash
python evaluate.py --checkpoint runs/tiny_sdv5_fusion_a_c/best.pt --dataset tiny-genimage --split val --generators sdv5 --forensic-size 224 --patch-size 32 --stride 16 --top-k 8
```

```bash
python evaluate.py --config configs/eval_cifake.json
```

```bash
python evaluate.py --config configs/eval_tiny_sdv5_fusion_a_c.json
```

## Recommended Hyperparameters

For CIFAKE:

- `semantic_size=224`
- `forensic_size=null`
- `patch_size=16`
- `stride=8`
- `top_k=4`

For Tiny-GenImage:

- `semantic_size=224`
- `forensic_size=224`
- `patch_size=32`
- `stride=16`
- `top_k=8`

## Files

- `src/data.py`: dataset scanning and paired semantic/forensic transforms
- `src/config.py`: JSON config loading and resolved config saving
- `src/branch_c.py`: FFT patch selector and patch-level forensic branch
- `src/fusion.py`: branch A + branch C fusion model
- `src/engine.py`: training, evaluation, checkpoint helpers
- `src/metrics.py`: accuracy, balanced accuracy, precision, recall, F1, AUROC
- `train.py`: training CLI
- `evaluate.py`: evaluation CLI
