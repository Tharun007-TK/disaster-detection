"""Siamese ResNet18 for pre/post damage segmentation.

Branches share weights from layer1 onward. First conv differs:
  - pre stem: 3-channel RGB (ImageNet pretrained)
  - post stem: 1-channel SAR (initialized from averaged pretrained RGB weights)

Output: per-pixel class logits [B, num_classes, H, W] at input resolution.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SharedResNet18Trunk(nn.Module):
    """ResNet18 trunk without conv1/bn1 — layer1..layer4, shared across branches."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 512, H/32, W/32]


class BranchStem(nn.Module):
    """conv1 + bn1 + relu + maxpool, per-branch (not shared)."""

    def __init__(self, in_channels: int, pretrained_stem: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if pretrained_stem is not None:
            self._init_from_pretrained(pretrained_stem, in_channels)

    def _init_from_pretrained(self, pretrained: nn.Module, in_channels: int) -> None:
        src_conv: nn.Conv2d = pretrained.conv1  # type: ignore[assignment]
        src_bn: nn.BatchNorm2d = pretrained.bn1  # type: ignore[assignment]
        with torch.no_grad():
            if in_channels == 3:
                self.conv1.weight.copy_(src_conv.weight)
            elif in_channels == 1:
                self.conv1.weight.copy_(src_conv.weight.mean(dim=1, keepdim=True))
            else:
                mean_w = src_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
            self.bn1.weight.copy_(src_bn.weight)
            self.bn1.bias.copy_(src_bn.bias)
            self.bn1.running_mean.copy_(src_bn.running_mean)
            self.bn1.running_var.copy_(src_bn.running_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x  # [B, 64, H/4, W/4]


class SiameseDamageNet(nn.Module):
    """Siamese ResNet18 with per-branch stems and shared trunk + diff-based seg head."""

    def __init__(
        self,
        num_classes: int = 4,
        pre_channels: int = 3,
        post_channels: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        ref_backbone = resnet18(weights=weights) if pretrained else None

        self.pre_stem = BranchStem(pre_channels, pretrained_stem=ref_backbone)
        self.post_stem = BranchStem(post_channels, pretrained_stem=ref_backbone)
        self.trunk = SharedResNet18Trunk(pretrained=pretrained)

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def _encode(self, stem: BranchStem, x: torch.Tensor) -> torch.Tensor:
        x = stem(x)
        x = self.trunk(x)
        return x

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        feat_pre = self._encode(self.pre_stem, pre)
        feat_post = self._encode(self.post_stem, post)
        diff = torch.abs(feat_pre - feat_post)
        logits = self.head(diff)
        logits = F.interpolate(
            logits, size=pre.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SiameseDamageNet(num_classes=4, pretrained=False).to(device)
    pre = torch.randn(2, 3, 512, 512, device=device)
    post = torch.randn(2, 1, 512, 512, device=device)
    out = model(pre, post)
    print(f"pre: {tuple(pre.shape)}  post: {tuple(post.shape)}  out: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M  trainable: {n_trainable/1e6:.2f}M")

    print(f"pre_stem.conv1 id:  {id(model.pre_stem.conv1)}")
    print(f"post_stem.conv1 id: {id(model.post_stem.conv1)}")
    print(f"trunk.layer1 id:    {id(model.trunk.layer1)} (shared — used by both branches via self.trunk)")
