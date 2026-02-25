"""
tests/test_nyuv2_data.py
-------------------------
Comprehensive tests for the NYUv2 data pipeline.

Tests data loading, transforms, label integrity, depth masking,
and collation — all using MOCK data (no download required).

Run: python -m pytest tests/test_nyuv2_data.py -v
"""

import os
import shutil
import pytest
import numpy as np
import torch
from pathlib import Path

# ============================================================================
# FIXTURES: Create mock NYUv2 data (no download needed)
# ============================================================================

MOCK_ROOT = Path("tests/_mock_nyuv2")
MOCK_H, MOCK_W = 288, 384
MOCK_NUM_TRAIN = 5
MOCK_NUM_VAL = 3


@pytest.fixture(scope="module", autouse=True)
def mock_nyuv2_data():
    """Create temporary mock NYUv2 .npy files for testing."""
    for split, count in [("train", MOCK_NUM_TRAIN), ("val", MOCK_NUM_VAL)]:
        for modality in ["image", "label", "depth", "normal"]:
            mod_dir = MOCK_ROOT / split / modality
            mod_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                if modality == "image":
                    # (H, W, 3) float32, range [0, 255]
                    data = np.random.rand(MOCK_H, MOCK_W, 3).astype(np.float32) * 255.0
                elif modality == "label":
                    # (H, W) int, range {-1, 0..12}
                    data = np.random.randint(-1, 13, size=(MOCK_H, MOCK_W)).astype(np.int32)
                elif modality == "depth":
                    # (H, W, 1) float32, some zeros (invalid)
                    data = np.random.rand(MOCK_H, MOCK_W, 1).astype(np.float32) * 5.0
                    # Inject 10% invalid depth pixels
                    mask = np.random.rand(MOCK_H, MOCK_W, 1) < 0.1
                    data[mask] = 0.0
                elif modality == "normal":
                    # (H, W, 3) float32, unit normals
                    data = np.random.randn(MOCK_H, MOCK_W, 3).astype(np.float32)
                    # Normalize to unit length
                    norms = np.linalg.norm(data, axis=-1, keepdims=True)
                    norms = np.maximum(norms, 1e-6)
                    data = data / norms

                np.save(str(mod_dir / f"{i}.npy"), data)

    yield str(MOCK_ROOT)

    # Cleanup
    if MOCK_ROOT.exists():
        shutil.rmtree(str(MOCK_ROOT))


# ============================================================================
# TEST 1: SHAPE VALIDATION
# ============================================================================

class TestShapes:
    """Verify all output tensors have correct shapes and dtypes."""

    def test_train_sample_shapes(self, mock_nyuv2_data):
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]

        assert sample["input"].shape == (3, MOCK_H, MOCK_W), \
            f"Image shape: {sample['input'].shape}"
        assert sample["targets"]["segmentation"].shape == (MOCK_H, MOCK_W), \
            f"Seg shape: {sample['targets']['segmentation'].shape}"
        assert sample["targets"]["depth"].shape == (1, MOCK_H, MOCK_W), \
            f"Depth shape: {sample['targets']['depth'].shape}"
        assert sample["targets"]["normals"].shape == (3, MOCK_H, MOCK_W), \
            f"Normal shape: {sample['targets']['normals'].shape}"

    def test_dtypes(self, mock_nyuv2_data):
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]

        assert sample["input"].dtype == torch.float32, "Image must be float32"
        assert sample["targets"]["segmentation"].dtype == torch.int64, \
            f"Segmentation MUST be int64 for CrossEntropyLoss, got {sample['targets']['segmentation'].dtype}"
        assert sample["targets"]["depth"].dtype == torch.float32, "Depth must be float32"
        assert sample["targets"]["normals"].dtype == torch.float32, "Normals must be float32"

    def test_val_sample_shapes(self, mock_nyuv2_data):
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="val", augmentation=False, validate_schema=False)
        assert len(ds) == MOCK_NUM_VAL

        sample = ds[0]
        assert sample["input"].shape == (3, MOCK_H, MOCK_W)

    def test_meta_fields_present(self, mock_nyuv2_data):
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]

        assert "meta" in sample
        assert "depth_mask" in sample["meta"]
        assert "sample_id" in sample["meta"]
        assert sample["meta"]["depth_mask"].shape == (1, MOCK_H, MOCK_W)
        assert isinstance(sample["meta"]["sample_id"], str)


# ============================================================================
# TEST 2: LABEL INTEGRITY
# ============================================================================

class TestLabelIntegrity:
    """Verify label remapping and value ranges."""

    def test_no_negative_labels(self, mock_nyuv2_data):
        """Labels must not contain -1 (should be remapped to 255)."""
        from spectra.data.nyuv2 import NYUv2Dataset, IGNORE_INDEX
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]
        labels = sample["targets"]["segmentation"]

        assert (labels >= 0).all(), "Found negative labels — -1 should be remapped to 255"

    def test_label_range_valid(self, mock_nyuv2_data):
        """All labels must be in {0..12} or 255 (ignore)."""
        from spectra.data.nyuv2 import NYUv2Dataset, NUM_CLASSES, IGNORE_INDEX
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]
        labels = sample["targets"]["segmentation"]

        valid = ((labels >= 0) & (labels < NUM_CLASSES)) | (labels == IGNORE_INDEX)
        assert valid.all(), \
            f"Invalid label values found: {labels[~valid].unique().tolist()}"


# ============================================================================
# TEST 3: DEPTH MASKING
# ============================================================================

class TestDepthMask:
    """Verify depth validity mask correctness."""

    def test_mask_zeros_at_invalid_depth(self, mock_nyuv2_data):
        """Depth mask must be 0 where depth is 0 (Kinect failure)."""
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]

        depth = sample["targets"]["depth"]
        mask = sample["meta"]["depth_mask"]

        # Where depth is 0, mask must be 0
        invalid_pixels = (depth == 0.0)
        assert (mask[invalid_pixels] == 0.0).all(), "Depth mask should be 0 at invalid depth pixels"

    def test_mask_ones_at_valid_depth(self, mock_nyuv2_data):
        """Depth mask must be 1 where depth > 0."""
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)
        sample = ds[0]

        depth = sample["targets"]["depth"]
        mask = sample["meta"]["depth_mask"]

        valid_pixels = (depth > 0.0)
        assert (mask[valid_pixels] == 1.0).all(), "Depth mask should be 1 at valid depth pixels"


# ============================================================================
# TEST 4: TRANSFORM CORRECTNESS
# ============================================================================

class TestTransforms:
    """Test spatial transforms preserve critical invariants."""

    def test_horizontal_flip_negates_normal_x(self):
        """Horizontal flip must negate ONLY normal[0] (x-component)."""
        from spectra.data.nyuv2.transforms import RandomHorizontalFlip

        # Force flip (p=1.0)
        flip = RandomHorizontalFlip(p=1.0)

        img = torch.randn(3, 10, 10)
        label = torch.randint(0, 13, (10, 10))
        depth = torch.rand(1, 10, 10)
        normal = torch.randn(3, 10, 10)

        orig_normal = normal.clone()
        _, _, _, flipped_normal = flip(img, label, depth, normal)

        # X-component should be negated (after spatial flip)
        # The spatial flip reverses the width axis, then x is negated
        expected_x = -torch.flip(orig_normal[0:1], dims=[2])
        actual_x = flipped_normal[0:1]
        assert torch.allclose(actual_x, expected_x, atol=1e-6), \
            "Normal x-component not correctly negated after flip"

        # Y and Z should just be spatially flipped (not negated)
        expected_yz = torch.flip(orig_normal[1:], dims=[2])
        actual_yz = flipped_normal[1:]
        assert torch.allclose(actual_yz, expected_yz, atol=1e-6), \
            "Normal y/z components should only be spatially flipped, not negated"

    def test_scale_crop_divides_depth(self):
        """RandomScaleCrop must divide depth by scale factor."""
        from spectra.data.nyuv2.transforms import RandomScaleCrop

        # Use single scale (1.5) so we know the exact divisor
        crop = RandomScaleCrop(scales=[1.5])

        img = torch.randn(3, 30, 30)
        label = torch.randint(0, 13, (30, 30))
        depth = torch.ones(1, 30, 30) * 3.0  # Constant depth = 3m
        normal = torch.randn(3, 30, 30)

        _, _, cropped_depth, _ = crop(img, label, depth, normal)

        # With nearest interpolation and constant input, output should be 3.0/1.5 = 2.0
        assert torch.allclose(cropped_depth, torch.ones_like(cropped_depth) * 2.0, atol=0.01), \
            f"Depth should be 3.0/1.5=2.0, got {cropped_depth.mean():.4f}"

    def test_scale_crop_preserves_label_integers(self):
        """RandomScaleCrop must use nearest interpolation for labels (no fractional class indices)."""
        from spectra.data.nyuv2.transforms import RandomScaleCrop

        crop = RandomScaleCrop(scales=[1.2])

        img = torch.randn(3, 30, 30)
        label = torch.randint(0, 13, (30, 30))
        depth = torch.rand(1, 30, 30)
        normal = torch.randn(3, 30, 30)

        _, cropped_label, _, _ = crop(img, label, depth, normal)

        # Labels must remain integer values (no fractional class indices from bilinear)
        assert cropped_label.dtype == torch.int64, f"Label dtype should be int64, got {cropped_label.dtype}"
        # All values should be exact integers present in the original
        unique_vals = cropped_label.unique()
        for v in unique_vals:
            assert v.item() in range(13) or v.item() == 255, \
                f"Invalid label value {v.item()} after crop — nearest interpolation failure"


# ============================================================================
# TEST 5: COLLATION
# ============================================================================

class TestCollation:
    """Verify custom collate function produces correct batch shapes."""

    def test_collate_batch(self, mock_nyuv2_data):
        from spectra.data.nyuv2 import NYUv2Dataset
        ds = NYUv2Dataset(root=mock_nyuv2_data, split="train", augmentation=False, validate_schema=False)

        samples = [ds[i] for i in range(min(3, len(ds)))]
        batch = NYUv2Dataset.collate_fn(samples)

        B = len(samples)
        assert batch["input"].shape == (B, 3, MOCK_H, MOCK_W)
        assert batch["targets"]["segmentation"].shape == (B, MOCK_H, MOCK_W)
        assert batch["targets"]["depth"].shape == (B, 1, MOCK_H, MOCK_W)
        assert batch["targets"]["normals"].shape == (B, 3, MOCK_H, MOCK_W)
        assert batch["meta"]["depth_mask"].shape == (B, 1, MOCK_H, MOCK_W)
        assert len(batch["meta"]["sample_id"]) == B


# ============================================================================
# TEST 6: DOWNLOAD UTILITY
# ============================================================================

class TestDownload:
    """Test download integrity verification (no actual download)."""

    def test_verify_integrity_passes_for_valid_mock(self, mock_nyuv2_data):
        from spectra.data.nyuv2.download import _verify_integrity
        # Our mock data has fewer files than expected (5 vs 795)
        # so this should FAIL — which is correct behavior
        assert not _verify_integrity(Path(mock_nyuv2_data))

    def test_verify_integrity_fails_for_missing_dir(self):
        from spectra.data.nyuv2.download import _verify_integrity
        assert not _verify_integrity(Path("nonexistent_dir_xyz"))
