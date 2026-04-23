"""
spectra/data/nyuv2/transforms.py
---------------------------------
Joint Spatial Transforms for NYUv2 Multi-Task Dense Prediction.

All transforms apply identical spatial operations to image + all task labels
to maintain pixel-level correspondence. This is CRITICAL — misaligned spatial
transforms between tasks silently poison the training manifold.

Design Decisions (verified against MTAN & LibMTL reference implementations):
1. Interpolation: bilinear for image/normals, nearest for segmentation/depth
2. Depth after scale-crop: divided by scale factor (metric depth preservation)
3. Normal x-flip: only channel 0 negated (x-axis direction)
4. All ops on (C, H, W) tensors (post np.moveaxis, matching MTAN convention)

References:
    - lorenmt/mtan (MTAN official): create_dataset.py
    - median-research-group/LibMTL: examples/nyu/create_dataset.py
    - PAD-Net (Liu et al. CVPR 2018)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# =============================================================================
# 1. RANDOM SCALE CROP (MTAN-Standard)
# =============================================================================

class RandomScaleCrop:
    """
    Multi-scale random crop for dense prediction augmentation.

    Randomly selects a scale factor, crops a correspondingly sized region,
    then resizes back to original dimensions. This simulates varying distance
    to the scene.

    CRITICAL DEPTH LOGIC:
        When scale > 1.0, we are "zooming in" — objects appear closer.
        Depth values must be divided by scale to maintain metric consistency.
        Example: scale=1.5 means we crop 1/1.5 of the image → objects are
        1.5× closer → depth should be depth/1.5.

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

        # Fast path: scale=1.0 is a no-op, skip expensive interpolation
        if sc == 1.0:
            return image, label, depth, normal

        # Crop region size (inverse of scale — larger scale = smaller crop)
        h, w = int(height / sc), int(width / sc)

        # Random crop position
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        # --- Image: bilinear interpolation (smooth RGB sub-pixel values) ---
        image_crop = F.interpolate(
            image[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0)

        # --- Segmentation: NEAREST interpolation (preserves class indices) ---
        # label is (H,W) → add batch+channel dims for F.interpolate → remove them
        label_crop = F.interpolate(
            label[None, None, i:i + h, j:j + w].float(),
            size=(height, width),
            mode='nearest',
        ).squeeze(0).squeeze(0).long()

        # --- Depth: NEAREST interpolation (preserves metric values) ---
        # CRITICAL: Divide by scale factor to maintain metric depth consistency
        depth_crop = F.interpolate(
            depth[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='nearest',
        ).squeeze(0)
        depth_crop = depth_crop / sc  # Metric depth preservation

        # --- Normals: bilinear interpolation (smooth directional field) ---
        normal_crop = F.interpolate(
            normal[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0)

        # Re-normalize normals after bilinear interpolation (interpolation breaks unit length)
        mag = normal_crop.norm(dim=0, keepdim=True)
        mag = mag.clamp(min=1e-8)
        normal_crop = normal_crop / mag

        return image_crop, label_crop, depth_crop, normal_crop


# =============================================================================
# 2. RANDOM HORIZONTAL FLIP
# =============================================================================

class RandomHorizontalFlip:
    """
    Joint horizontal flip for all modalities.

    CRITICAL NORMAL LOGIC:
        Surface normals encode 3D direction (x, y, z).
        Flipping the image horizontally mirrors the x-axis:
        → normal[0] (x-component) must be negated
        → normal[1] (y-component) stays the same
        → normal[2] (z-component) stays the same

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
            # Flip along width axis (dim=2 for (C,H,W), dim=1 for (H,W))
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
            depth = torch.flip(depth, dims=[2])
            normal = torch.flip(normal, dims=[2])

            # CRITICAL: Negate x-component of surface normals after spatial flip
            # torch.flip returns a view — clone channel 0 before negating to avoid aliasing
            normal = normal.clone()
            normal[0, :, :] = -normal[0, :, :]

        return image, label, depth, normal


# =============================================================================
# 3. IMAGENET NORMALIZATION (Optional)
# =============================================================================

class ImageNetNormalize:
    """
    Standard ImageNet channel normalization.

    Only applied when using pretrained backbones (ResNet, SegNet pretrained).
    MTAN baselines do NOT use this — they operate on raw [0, 255] float values.

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


# =============================================================================
# 4. COMPOSED TRANSFORMS
# =============================================================================

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
