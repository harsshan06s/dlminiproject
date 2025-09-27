"""Model factory built on top of timm."""

from __future__ import annotations

import timm
import torch.nn as nn


def create_model(backbone_name: str = "tf_efficientnet_b0", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Instantiate a timm model and return it."""
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    return model
