"""
Joint spatial transforms for NYUv2 dense prediction.

Each transform applies the same spatial operation to the image and all target
maps so that pixel-level correspondence is preserved.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RandomScaleCrop:
    """
    Multi-scale random crop for dense prediction augmentation.

    Randomly selects a scale factor, crops a correspondingly sized region,
    then resizes back to original dimensions. This simulates varying distance
    to the scene.

    Args:
        scales: List of allowed scale factors. Default [1.0, 1.2, 1.5]
                matches MTAN/PAD-Net standard.
    """

    def __init__(self, scales: Optional[List[float]] = None):
        self.scales = scales or [1.0, 1.2, 1.5]

    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image:  (3, H, W) float32 — RGB
            label:  (H, W)    long    — semantic class indices
            depth:  (1, H, W) float32 — metric depth (meters)
            normal: (3, H, W) float32 — surface normal (x, y, z)

        Returns:
            Tuple of (image, label, depth, normal) with same shapes.
        """
        height, width = image.shape[-2:]
        sc = random.choice(self.scales)

        # Skip interpolation when scale is unchanged.
        if sc == 1.0:
            return image, label, depth, normal

        # Larger scales correspond to smaller crop regions.
        h, w = int(height / sc), int(width / sc)

        # Random crop position
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        #  Image: bilinear interpolation (smooth RGB sub-pixel values) 
        image_crop = F.interpolate(
            image[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0)

        # Nearest interpolation preserves segmentation class indices.
        label_crop = F.interpolate(
            label[None, None, i:i + h, j:j + w].float(),
            size=(height, width),
            mode='nearest',
        ).squeeze(0).squeeze(0).long()

        # Divide by the scale factor to preserve metric depth.
        depth_crop = F.interpolate(
            depth[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='nearest',
        ).squeeze(0)
        depth_crop = depth_crop / sc  # Metric depth preservation

        # Re-normalize normals after interpolation.
        normal_crop = F.interpolate(
            normal[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0)

        mag = normal_crop.norm(dim=0, keepdim=True)
        mag = mag.clamp(min=1e-8)
        normal_crop = normal_crop / mag

        return image_crop, label_crop, depth_crop, normal_crop


class RandomHorizontalFlip:
    """
    Joint horizontal flip for all modalities.

    Args:
        p: Probability of flip. Default 0.5 matches MTAN.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            # Flip along the width axis.
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
            depth = torch.flip(depth, dims=[2])
            normal = torch.flip(normal, dims=[2])

            # Negate the x-component of surface normals after the spatial flip.
            normal = normal.clone()
            normal[0, :, :] = -normal[0, :, :]

        return image, label, depth, normal


class ImageNetNormalize:
    """
    Standard ImageNet channel normalization.

    Only applied when using pretrained backbones (ResNet, SegNet pretrained).
    MTAN baselines operate on raw [0, 255] float values.

    This is intentionally a separate class (not baked into the dataset) so
    ablation studies can toggle it cleanly.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (3, H, W) float32, expected range [0, 1] or [0, 255].

        Returns:
            Normalized image (3, H, W).
        """
        # If image is in [0, 255] range, scale to [0, 1] first
        if image.max() > 1.0:
            image = image / 255.0

        return TF.normalize(image, mean=self.MEAN, std=self.STD)


# 4. COMPOSED TRANSFORMS


class NYUv2TrainTransform:
    """
    Full training transform pipeline for NYUv2.

    Pipeline: RandomScaleCrop → RandomHorizontalFlip → (optional) ImageNetNormalize

    Args:
        scales: Scale factors for RandomScaleCrop.
        flip_p: Probability of horizontal flip.
        normalize_rgb: Whether to apply ImageNet normalization.
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        flip_p: float = 0.5,
        normalize_rgb: bool = False,
    ):
        self.scale_crop = RandomScaleCrop(scales)
        self.flip = RandomHorizontalFlip(p=flip_p)
        self.normalize = ImageNetNormalize() if normalize_rgb else None

    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label, depth, normal = self.scale_crop(image, label, depth, normal)
        image, label, depth, normal = self.flip(image, label, depth, normal)

        if self.normalize is not None:
            image = self.normalize(image)

        return image, label, depth, normal


class NYUv2BatchTrainTransform:
    """
    Batch-level NYUv2 augmentation for execution after device transfer.

    Uses F.affine_grid + F.grid_sample to apply per-sample random scale-crop
    and horizontal flip in a single batched CUDA operation, eliminating the
    per-sample loop that would otherwise issue N separate kernel launches.

    Coordinate math (align_corners=True convention):
        Crop region [i:i+h, j:j+w] in input maps to full output [0:H, 0:W].
        Affine theta:
            sx = w_crop / W          sy = h_crop / H
            tx = (2*j + w_crop)/W - 1   ty = (2*i + h_crop)/H - 1
        Horizontal flip integrated by negating sx and tx.
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        flip_p: float = 0.5,
        normalize_rgb: bool = False,
    ):
        self.scales = scales or [1.0, 1.2, 1.5]
        self.flip_p = flip_p
        self.normalize = ImageNetNormalize() if normalize_rgb else None

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = batch["input"]                           # (B, 3, H, W)
        label = batch["targets"]["segmentation"]          # (B, H, W) long
        depth = batch["targets"]["depth"]                 # (B, 1, H, W) float
        normal = batch["targets"]["normals"]              # (B, 3, H, W) float

        B, _, H, W = image.shape
        device = image.device

        #  1. Per-sample random parameters (vectorized) 
        scale_idx = torch.randint(len(self.scales), (B,), device=device)
        scales_b = torch.tensor(self.scales, device=device, dtype=torch.float32)[scale_idx]  # (B,)
        flip_mask = torch.rand(B, device=device) < self.flip_p                              # (B,)

        #  2. Crop geometry 
        h_crop = (H / scales_b).long().clamp(max=H)       # (B,)
        w_crop = (W / scales_b).long().clamp(max=W)       # (B,)

        max_i = (H - h_crop).clamp(min=0)
        max_j = (W - w_crop).clamp(min=0)
        i_off = (torch.rand(B, device=device) * (max_i + 1).float()).long().clamp(max=max_i)
        j_off = (torch.rand(B, device=device) * (max_j + 1).float()).long().clamp(max=max_j)

        #  3. Build affine theta (B, 2, 3) 
        sx = w_crop.float() / W
        sy = h_crop.float() / H
        tx = (2.0 * j_off.float() + w_crop.float()) / W - 1.0
        ty = (2.0 * i_off.float() + h_crop.float()) / H - 1.0

        # Integrate horizontal flip: negate sx and tx for flipped samples
        flip_sign = torch.where(flip_mask, -1.0, 1.0).to(dtype=sx.dtype)
        sx = sx * flip_sign
        tx = tx * flip_sign

        zeros = torch.zeros_like(sx)
        theta = torch.stack([
            torch.stack([sx, zeros, tx], dim=1),
            torch.stack([zeros, sy, ty], dim=1),
        ], dim=1)  # (B, 2, 3)

        #  4. Batched grid_sample 
        grid = F.affine_grid(theta, image.shape, align_corners=True)  # (B, H, W, 2)

        # Image: bilinear
        image = F.grid_sample(image, grid, mode='bilinear', align_corners=True, padding_mode='border')

        # Label: nearest (add channel dim, convert back to long)
        label = F.grid_sample(
            label.unsqueeze(1).float(), grid, mode='nearest', align_corners=True, padding_mode='border'
        ).squeeze(1).to(dtype=torch.long)

        # Depth: nearest + metric scale correction
        depth = F.grid_sample(depth, grid, mode='nearest', align_corners=True, padding_mode='border')
        depth = depth / scales_b.view(B, 1, 1, 1)

        # Normals: bilinear + renormalize
        normal = F.grid_sample(normal, grid, mode='bilinear', align_corners=True, padding_mode='border')
        mag = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal = normal / mag

        # Negate x-component of normals for horizontally flipped samples
        if flip_mask.any():
            flip_idx = flip_mask.nonzero(as_tuple=True)[0]
            normal[flip_idx, 0, :, :] = -normal[flip_idx, 0, :, :]

        #  5. Write back 
        batch["input"] = image
        batch["targets"]["segmentation"] = label
        batch["targets"]["depth"] = depth
        batch["targets"]["normals"] = normal
        batch["meta"]["depth_mask"] = (depth > 0.0).float()

        if self.normalize is not None:
            batch["input"] = self.normalize(batch["input"])

        return batch


class NYUv2TestTransform:
    """
    Test/validation transform — no augmentation.

    Optionally applies ImageNet normalization for pretrained backbone compatibility.

    Args:
        normalize_rgb: Whether to apply ImageNet normalization.
    """

    def __init__(self, normalize_rgb: bool = False):
        self.normalize = ImageNetNormalize() if normalize_rgb else None

    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalize is not None:
            image = self.normalize(image)

        return image, label, depth, normal
