from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from src.config import parse_args_with_config
from src.data import PairedTransform, build_dataset
from src.engine import evaluate, evaluate_by_generator, load_model_weights
from src.branch_c import PatchForensicBranch
from src.fusion import FusionForensicDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Branch A + Branch C fusion checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--dataset", choices=["cifake", "tiny-genimage"], default="cifake")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--generators", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--semantic-size", type=int, default=224)
    parser.add_argument("--forensic-size", type=int, default=None)
    parser.add_argument("--branch-a-backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--branch-a-feature-dim", type=int, default=128)
    parser.add_argument("--pretrained-branch-a", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-branch-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--branch-c-feature-dim", "--feature-dim", dest="branch_c_feature_dim", type=int, default=128)
    parser.add_argument("--freeze-branch-c", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args_with_config(parser)
    if not args.checkpoint:
        parser.error("--checkpoint is required, either in config JSON or on the command line.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=False,
        augment=False,
    )
    dataset = build_dataset(args.dataset_root, args.dataset, args.split, transform, generators=args.generators)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    branch_c = PatchForensicBranch(
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_dim=args.branch_c_feature_dim,
    )
    model = FusionForensicDetector(
        branch_c=branch_c,
        branch_a_backbone=args.branch_a_backbone,
        branch_a_feature_dim=args.branch_a_feature_dim,
        branch_c_feature_dim=args.branch_c_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        pretrained_branch_a=args.pretrained_branch_a,
        freeze_branch_a=args.freeze_branch_a,
        freeze_branch_c=args.freeze_branch_c,
    ).to(device)
    checkpoint = load_model_weights(args.checkpoint, model, device)
    metrics = evaluate(model, loader, device)
    result = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "overall": metrics,
        "by_generator": evaluate_by_generator(model, loader, device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
