from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


class SemanticBranchA(nn.Module):
    """
    Branch A: Semantic / Global Feature Branch.

    This branch extracts global semantic features from the normalized semantic view.
    Expected input:
        images: Tensor with shape (B, 3, H, W), usually batch["image_semantic"]

    Output:
        feature: Tensor with shape (B, feature_dim)
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        feature_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.feature_dim = feature_dim

        backbone_model, backbone_out_dim = self._build_backbone(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.backbone = backbone_model

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: semantic image tensor, shape (B, 3, H, W)

        Returns:
            semantic feature tensor, shape (B, feature_dim)
        """
        backbone_feature = self.backbone(images)
        feature = self.projection(backbone_feature)
        return feature

    @staticmethod
    def _build_backbone(
        backbone: str,
        pretrained: bool,
    ) -> tuple[nn.Module, int]:
        """
        Build a torchvision ResNet backbone and remove its final classifier.
        """

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            out_dim = model.fc.in_features

        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            out_dim = model.fc.in_features

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            out_dim = model.fc.in_features

        else:
            raise ValueError(
                f"Unsupported Branch A backbone: {backbone}. "
                "Supported options are: resnet18, resnet34, resnet50."
            )

        model.fc = nn.Identity()
        return model, out_dim


class BranchAClassifier(nn.Module):
    """
    Optional standalone classifier for Branch A.

    This is useful for ablation experiments:
        image_semantic -> Branch A -> classifier -> real/fake logits

    Main fusion model does not have to use this class.
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        feature_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = SemanticBranchA(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(images)
        logits = self.classifier(feature).squeeze(1)

        return {
            "logits": logits,
            "features": feature,
        }