"""Polyp dataset with point-prompt sampling from GT masks.

GT masks are ONLY used for:
  a) generating point prompts
  b) validation/test metrics

GT masks are NEVER used in training loss.
"""

import os
import json
import random
import hashlib
import tempfile
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _name_hash(name):
    """Deterministic hash of a string → int in [0, 9999]."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10000


def _find_prior_path(prior_dir, dataset_name, sample_name):
    if not prior_dir:
        return None
    stem = os.path.splitext(sample_name)[0]
    search_dirs = [prior_dir]
    if dataset_name:
        search_dirs.insert(0, os.path.join(prior_dir, dataset_name))
    for root in search_dirs:
        for ext in ('.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
            candidate = os.path.join(root, stem + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def _prior_array_to_image(prior_arr):
    prior_arr = np.asarray(prior_arr, dtype=np.float32)
    if prior_arr.ndim == 3:
        if prior_arr.shape[0] == 1:
            prior_arr = prior_arr[0]
        elif prior_arr.shape[0] == 3 and prior_arr.shape[-1] != 3:
            prior_arr = prior_arr.mean(axis=0)
        elif prior_arr.shape[-1] == 1:
            prior_arr = prior_arr[..., 0]
        else:
            prior_arr = prior_arr.mean(axis=-1)
    if prior_arr.ndim != 2:
        raise ValueError(f"Expected 2D SAM prior, got shape {prior_arr.shape}")
    prior_arr = np.nan_to_num(prior_arr, nan=0.0, posinf=1.0, neginf=0.0)
    if prior_arr.size > 0 and float(np.max(prior_arr)) > 1.0:
        prior_arr = prior_arr / 255.0
    prior_arr = np.clip(prior_arr, 0.0, 1.0)
    return Image.fromarray((prior_arr * 255.0).astype(np.uint8), mode='L')


# ---------------------------------------------------------------------------
# Split management
# ---------------------------------------------------------------------------

def generate_split(data_root, dataset_name, split_dir, train_ratio=0.8,
                   test_ratio=0.05, seed=42):
    """Generate a fixed train/test split for one dataset and save to JSON.

    Sorts filenames, shuffles with fixed seed, splits by ratio.  Overwrites
    an existing split file only when explicitly requested via --regenerate_splits.
    """
    img_dir = os.path.join(data_root, dataset_name, 'images')
    mask_dir = os.path.join(data_root, dataset_name, 'masks')

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_names = sorted(os.listdir(img_dir))
    image_paths = [os.path.join(img_dir, n) for n in image_names]

    # Match masks: assume same filename (may differ in extension)
    mask_paths = []
    valid_images = []
    for img_path, name in zip(image_paths, image_names):
        base, _ = os.path.splitext(name)
        # Try common extensions
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            candidate = os.path.join(mask_dir, base + ext)
            if os.path.exists(candidate):
                mask_paths.append(candidate)
                valid_images.append(img_path)
                break
        else:
            # Try exact name match
            candidate = os.path.join(mask_dir, name)
            if os.path.exists(candidate):
                mask_paths.append(candidate)
                valid_images.append(img_path)

    if len(valid_images) == 0:
        raise RuntimeError(f"No image-mask pairs found in {img_dir} / {mask_dir}")

    # Sort and shuffle
    pairs = sorted(zip(valid_images, mask_paths), key=lambda x: os.path.basename(x[0]))
    rng = random.Random(seed)
    rng.shuffle(pairs)

    N = len(pairs)
    n_test = max(1, int(round(N * test_ratio)))
    n_test = min(n_test, N - 1)
    n_train = N - n_test
    effective_train_ratio = n_train / float(N)
    effective_test_ratio = n_test / float(N)

    split_data = {
        'dataset': dataset_name,
        'seed': seed,
        'train_ratio': effective_train_ratio,
        'test_ratio': effective_test_ratio,
        'num_samples': N,
        'train': [{'image': p[0], 'mask': p[1], 'name': os.path.basename(p[0])} for p in pairs[:n_train]],
        'test':  [{'image': p[0], 'mask': p[1], 'name': os.path.basename(p[0])} for p in pairs[n_train:]],
    }

    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f'{dataset_name}_split_seed{seed}.json')
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{dataset_name}_split_seed{seed}_",
        suffix=".json.tmp",
        dir=split_dir,
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(split_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, split_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return split_data


def discover_samples(data_root, dataset_name):
    """Return all image/mask pairs for a dataset without creating a split."""
    img_dir = os.path.join(data_root, dataset_name, 'images')
    mask_dir = os.path.join(data_root, dataset_name, 'masks')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    samples = []
    for name in sorted(os.listdir(img_dir)):
        image_path = os.path.join(img_dir, name)
        if not os.path.isfile(image_path):
            continue
        stem, _ = os.path.splitext(name)
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.isfile(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            candidate = os.path.join(mask_dir, name)
            if os.path.isfile(candidate):
                mask_path = candidate
        if mask_path is not None:
            samples.append({'image': image_path, 'mask': mask_path, 'name': name})

    if not samples:
        raise RuntimeError(f"No image-mask pairs found in {img_dir} / {mask_dir}")
    return samples


def load_split(dataset_name, split_dir, seed=42):
    """Load an existing split file. Returns None if not found.

    Old split files may contain a `val` split. To refactor the pipeline to
    train/test only without wasting data, those samples are merged into train.
    No sample is duplicated across train/test.
    """
    split_path = os.path.join(split_dir, f'{dataset_name}_split_seed{seed}.json')
    if not os.path.exists(split_path):
        return None
    with open(split_path, 'r') as f:
        split = json.load(f)
    if 'val' in split:
        split['train'] = list(split.get('train', [])) + list(split.get('val', []))
        split.pop('val', None)
        split['num_samples'] = len(split['train']) + len(split.get('test', []))
    return split


# ---------------------------------------------------------------------------
# Point sampling helpers
# ---------------------------------------------------------------------------

def _sample_points_from_mask(mask, num_points, min_distance, max_attempts=100):
    """Sample points from a binary mask with minimum distance constraint.

    Args:
        mask: [H, W] binary numpy array (foreground region)
        num_points: target number of points
        min_distance: minimum Euclidean distance between any two points
        max_attempts: max random tries per point

    Returns:
        list of (y, x) tuples
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    indices = np.arange(len(ys))
    points = []

    for _ in range(num_points):
        for _ in range(max_attempts):
            idx = np.random.choice(indices)
            py, px = ys[idx], xs[idx]
            # Check distance to existing points
            too_close = False
            for ey, ex in points:
                if ((py - ey) ** 2 + (px - ex) ** 2) ** 0.5 < min_distance:
                    too_close = True
                    break
            if not too_close:
                points.append((py, px))
                break

    return points


def _render_point_map(points, H, W, radius, valid_mask=None):
    """Render points as filled disks on a [H, W] map.

    Args:
        points: list of (y, x) tuples
        H, W: spatial dimensions
        radius: disk radius in pixels
        valid_mask: optional boolean/0-1 mask restricting where disks may occupy

    Returns:
        numpy float32 array [H, W]
    """
    pmap = np.zeros((H, W), dtype=np.float32)
    if not points:
        return pmap

    Y, X = np.ogrid[:H, :W]
    for py, px in points:
        dist = np.sqrt((Y - py) ** 2 + (X - px) ** 2)
        disk = dist <= radius
        if valid_mask is not None:
            disk = disk & valid_mask.astype(bool)
        pmap[disk] = 1.0
    return pmap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _pack_point_tensors(fg_points, bg_points, max_fg_points, max_bg_points):
    """Pack variable sampled points into fixed-size tensors for batching."""
    total_points = max_fg_points + max_bg_points
    coords = torch.full((total_points, 2), -1, dtype=torch.long)
    labels = torch.full((total_points,), -1, dtype=torch.long)

    for idx, (py, px) in enumerate(fg_points[:max_fg_points]):
        coords[idx] = torch.tensor([py, px], dtype=torch.long)
        labels[idx] = 1

    base = max_fg_points
    for idx, (py, px) in enumerate(bg_points[:max_bg_points]):
        coords[base + idx] = torch.tensor([py, px], dtype=torch.long)
        labels[base + idx] = 0

    return coords, labels


class PolypDataset(Dataset):
    """Single-dataset polyp loader with point-prompt sampling.

    Args:
        samples: list of dicts with 'image', 'mask', 'name' keys
        image_size: resize to (image_size, image_size)
        num_fg_points: exact number of foreground points to sample
        num_bg_points: exact number of background points to sample
        min_point_distance: minimum pixel distance between sampled points
        point_radius: disk radius for rendering point maps
        is_train: if True, random point sampling; if False, fixed sampling
        seed: base seed for reproducible point sampling
    """

    def __init__(self, samples, image_size=352, num_fg_points=1, num_bg_points=1,
                 min_point_distance=20, point_radius=10, is_train=True, seed=42,
                 test_noise_std=0.0, sam_prior_dir=None, dataset_name=None,
                 sam_prior_required=False):
        self.samples = samples
        self.image_size = image_size
        self.num_fg_points = num_fg_points
        self.num_bg_points = num_bg_points
        self.min_point_distance = min_point_distance
        self.point_radius = point_radius
        self.is_train = is_train
        self.seed = seed
        self.epoch = 0
        self.test_noise_std = float(test_noise_std)
        self.sam_prior_dir = sam_prior_dir
        self.dataset_name = dataset_name
        self.sam_prior_required = bool(sam_prior_required)

    def set_epoch(self, epoch):
        """Set epoch for reproducible point sampling (training only)."""
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def _augmentation_seed(self, sample_name):
        sample_hash = _name_hash(sample_name)
        return self.seed + self.epoch * 10000 + sample_hash + 314159

    def _load_sam_prior(self, sample_name):
        prior_path = _find_prior_path(self.sam_prior_dir, self.dataset_name, sample_name)
        if prior_path is None:
            if self.sam_prior_required:
                raise FileNotFoundError(f"SAM prior not found for sample '{sample_name}' in {self.sam_prior_dir}")
            return Image.new('L', (self.image_size, self.image_size), 0)
        if prior_path.endswith('.npy'):
            prior = _prior_array_to_image(np.load(prior_path))
        else:
            prior = Image.open(prior_path).convert('L')
        return prior.resize((self.image_size, self.image_size), Image.BILINEAR)

    def _apply_train_augmentation(self, image, mask, sample_name, sam_prior=None):
        """Apply train-only joint augmentation to image and mask.

        Geometric transforms are always applied jointly so the later point-map
        generation is consistent with the augmented image/mask/prior tuple.
        """
        rng = random.Random(self._augmentation_seed(sample_name))

        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if sam_prior is not None:
                sam_prior = sam_prior.transpose(Image.FLIP_LEFT_RIGHT)

        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if sam_prior is not None:
                sam_prior = sam_prior.transpose(Image.FLIP_TOP_BOTTOM)

        rot_k = rng.randint(0, 3)
        if rot_k:
            angle = 90 * rot_k
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
            if sam_prior is not None:
                sam_prior = sam_prior.rotate(angle, resample=Image.BILINEAR)

        image_np = np.array(image, dtype=np.float32)
        brightness = 1.0 + rng.uniform(-0.10, 0.10)
        contrast = 1.0 + rng.uniform(-0.10, 0.10)
        image_np = np.clip(image_np * brightness, 0.0, 255.0)
        image_np = np.clip((image_np - 127.5) * contrast + 127.5, 0.0, 255.0)
        image = Image.fromarray(image_np.astype(np.uint8))

        return image, mask, sam_prior

    def _apply_test_noise(self, image_np, sample_name):
        if self.is_train or self.test_noise_std <= 0:
            return image_np
        sample_hash = _name_hash(sample_name)
        rng = np.random.default_rng(self.seed + sample_hash)
        noise = rng.normal(0.0, self.test_noise_std, size=image_np.shape).astype(np.float32)
        return np.clip(image_np + noise, 0.0, 1.0)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -- Load image --
        image = Image.open(sample['image']).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(sample['mask']).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        sam_prior = self._load_sam_prior(sample['name'])

        if self.is_train:
            image, mask, sam_prior = self._apply_train_augmentation(image, mask, sample['name'], sam_prior=sam_prior)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = self._apply_test_noise(image_np, sample['name'])
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]

        # -- Load mask (for point generation + metrics only, NOT for loss) --
        mask_np = np.array(mask, dtype=np.float32)
        # Binarise
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]
        sam_prior_np = np.array(sam_prior, dtype=np.float32) / 255.0
        sam_prior_tensor = torch.from_numpy(np.clip(sam_prior_np, 0.0, 1.0)).unsqueeze(0)

        # -- Generate point maps --
        point_maps, point_coords, point_labels = self._generate_point_maps(mask_np, sample['name'])
        # point_maps: [2, H, W]  ch0=fg, ch1=bg

        return {
            'image': image_tensor,
            'point_maps': point_maps,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask': mask_tensor,
            'sam_prior': sam_prior_tensor,
            'name': sample['name'],
        }

    def _generate_point_maps(self, mask_np, sample_name):
        """Generate foreground and background point maps from GT mask.

        Training: random sampling per-epoch with per-sample variation.
        Val/Test: deterministic sampling per sample (epoch-independent).
        """
        H, W = mask_np.shape

        # Per-sample deterministic base (cross-run stable via MD5)
        sample_hash = _name_hash(sample_name)

        # Save/restore global RNG state to avoid side effects
        rng_state = random.getstate()
        np_state = np.random.get_state()

        if self.is_train:
            seed_val = self.seed + self.epoch * 10000 + sample_hash
        else:
            # Val/test: fixed per-sample seeding (epoch-independent)
            seed_val = self.seed + sample_hash

        random.seed(seed_val)
        np.random.seed(seed_val)

        fg_mask = mask_np > 0.5
        bg_mask = ~fg_mask

        fg_points = _sample_points_from_mask(fg_mask, self.num_fg_points, self.min_point_distance)
        bg_points = _sample_points_from_mask(bg_mask, self.num_bg_points, self.min_point_distance)

        random.setstate(rng_state)
        np.random.set_state(np_state)

        fg_map = _render_point_map(fg_points, H, W, self.point_radius, valid_mask=fg_mask)
        bg_map = _render_point_map(bg_points, H, W, self.point_radius, valid_mask=bg_mask)

        # Safety: numerical or boundary edge cases should never produce overlap.
        overlap = (fg_map > 0) & (bg_map > 0)
        if overlap.any():
            fg_map[overlap] = 0.0
            bg_map[overlap] = 0.0

        point_maps = np.stack([fg_map, bg_map], axis=0)  # [2, H, W]
        point_coords, point_labels = _pack_point_tensors(
            fg_points,
            bg_points,
            self.num_fg_points,
            self.num_bg_points,
        )
        return torch.from_numpy(point_maps), point_coords, point_labels


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(dataset_name, data_root, split_dir, image_size=352,
                       batch_size=8, num_workers=4, num_fg_points=1, num_bg_points=1,
                       min_point_distance=20, point_radius=10, seed=42,
                       train_ratio=0.8, test_ratio=0.05,
                       test_noise_std=0.0,
                       sam_prior_dir=None, sam_prior_required=False,
                       regenerate_splits=False):
    """Create train/test dataloaders for a single dataset.

    Returns:
        train_loader, test_loader
    """
    split = load_split(dataset_name, split_dir, seed)
    expected_test_ratio = float(test_ratio)
    actual_test_ratio = None if split is None else float(split.get('test_ratio', -1.0))
    ratio_mismatch = (
        split is not None
        and actual_test_ratio >= 0.0
        and abs(actual_test_ratio - expected_test_ratio) > 1e-6
    )
    if split is None or regenerate_splits or ratio_mismatch:
        split = generate_split(data_root, dataset_name, split_dir,
                               train_ratio, test_ratio, seed)

    train_ds = PolypDataset(split['train'], image_size=image_size,
                            num_fg_points=num_fg_points, num_bg_points=num_bg_points,
                            min_point_distance=min_point_distance,
                            point_radius=point_radius, is_train=True, seed=seed,
                            sam_prior_dir=sam_prior_dir, dataset_name=dataset_name,
                            sam_prior_required=sam_prior_required)

    test_ds = PolypDataset(split['test'], image_size=image_size,
                           num_fg_points=num_fg_points, num_bg_points=num_bg_points,
                           min_point_distance=min_point_distance,
                           point_radius=point_radius, is_train=False, seed=seed,
                           test_noise_std=test_noise_std,
                           sam_prior_dir=sam_prior_dir, dataset_name=dataset_name,
                           sam_prior_required=sam_prior_required)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
