"""Shared Point2Mask OT utilities used by the V4 branch."""

from heapq import heappop, heappush
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_PYDIJKSTRA_DIR = Path(__file__).resolve().parents[1] / 'Point2Mask-main' / 'easymd' / 'ops' / 'py-dijkstra' / 'pydijkstra'
if _PYDIJKSTRA_DIR.exists():
    sys.path.insert(0, str(_PYDIJKSTRA_DIR))
try:
    from pydijkstra import dijkstra_image as official_dijkstra_image
except Exception:
    official_dijkstra_image = None


_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_OFFSETS_4 = [
    (-1, 0),
    (0, -1), (0, 1),
    (1, 0),
]


def _get_offsets(neighborhood):
    if neighborhood == 4:
        return _OFFSETS_4
    if neighborhood == 8:
        return _OFFSETS_8
    raise ValueError(f"Unsupported neighborhood: {neighborhood}")


def _offset_step_lengths(offsets, device=None, dtype=None):
    lengths = []
    for dy, dx in offsets:
        if dy == 0 or dx == 0:
            lengths.append(1.0)
        else:
            lengths.append(float(np.sqrt(2.0)))
    if device is None and dtype is None:
        return lengths
    return torch.tensor(lengths, device=device, dtype=dtype)


def _normalize_map(x):
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    return x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-6)


def _stabilize_tensor(x, clamp_value=20.0):
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    return x.clamp(min=-clamp_value, max=clamp_value)


def _build_connected_area_mask(score_map, target_ratio, fg_point_mask=None, bg_point_mask=None):
    h, w = score_map.shape
    total = h * w
    k = max(1, int(round(float(target_ratio) * total)))

    fg_seed = (fg_point_mask > 0) if fg_point_mask is not None else torch.zeros((h, w), dtype=torch.bool, device=score_map.device)
    bg_block = (bg_point_mask > 0) if bg_point_mask is not None else torch.zeros((h, w), dtype=torch.bool, device=score_map.device)

    allowed = ~bg_block
    hard_mask = torch.zeros((h, w), dtype=torch.bool, device=score_map.device)
    hard_mask[fg_seed] = True

    if int(hard_mask.sum().item()) >= k:
        out = hard_mask.float()
        out[bg_block] = 0.0
        return out

    score_np = score_map.detach().cpu().numpy()
    allowed_np = allowed.detach().cpu().numpy()
    selected_np = hard_mask.detach().cpu().numpy()
    fg_seed_np = fg_seed.detach().cpu().numpy()

    heap = []
    visited = set()
    for sy, sx in np.argwhere(fg_seed_np):
        for dy, dx in _OFFSETS_8:
            ny, nx = sy + dy, sx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not allowed_np[ny, nx]:
                continue
            key = (int(ny), int(nx))
            if key in visited:
                continue
            visited.add(key)
            heappush(heap, (-float(score_np[ny, nx]), int(ny), int(nx)))

    current = int(selected_np.sum())
    while current < k and heap:
        _, y, x = heappop(heap)
        if selected_np[y, x] or not allowed_np[y, x]:
            continue
        selected_np[y, x] = True
        current += 1
        for dy, dx in _OFFSETS_8:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not allowed_np[ny, nx]:
                continue
            key = (int(ny), int(nx))
            if key in visited:
                continue
            visited.add(key)
            heappush(heap, (-float(score_np[ny, nx]), int(ny), int(nx)))

    if current < k:
        flat_score = score_map.masked_fill(~allowed, -1e6).reshape(-1)
        topk_idx = torch.topk(flat_score, k=min(k, total), largest=True).indices
        fallback_mask = torch.zeros_like(flat_score, dtype=torch.bool)
        fallback_mask[topk_idx] = True
        selected_np = fallback_mask.view(h, w).detach().cpu().numpy() | selected_np

    hard_mask = torch.from_numpy(selected_np.astype(np.float32)).to(score_map.device)
    hard_mask[fg_seed] = 1.0
    hard_mask[bg_block] = 0.0
    return hard_mask


def _compute_image_edge(image):
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    kernel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return _normalize_map(edge)


def _neighbour_feature_cost(feature_map, offsets=None):
    offsets = offsets or _OFFSETS_8
    feat = F.normalize(feature_map, dim=0)
    costs = []
    h, w = feat.shape[-2:]
    for dy, dx in offsets:
        shifted = torch.roll(feat, shifts=(dy, dx), dims=(-2, -1))
        dot = (feat * shifted).sum(0).clamp(-1.0, 1.0)
        diff = (1.0 - dot) * 0.5
        if dy < 0:
            diff[h + dy:, :] = diff.max()
        elif dy > 0:
            diff[:dy, :] = diff.max()
        if dx < 0:
            diff[:, w + dx:] = diff.max()
        elif dx > 0:
            diff[:, :dx] = diff.max()
        costs.append(diff)
    return torch.stack(costs, dim=-1)


def _neighbour_scalar_cost(value_map, mode="abs", offsets=None):
    offsets = offsets or _OFFSETS_8
    costs = []
    h, w = value_map.shape[-2:]
    for dy, dx in offsets:
        shifted = torch.roll(value_map, shifts=(dy, dx), dims=(-2, -1))
        if mode == "abs":
            diff = (value_map - shifted).abs()
        elif mode == "max":
            diff = torch.maximum(value_map, shifted)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if dy < 0:
            diff[h + dy:, :] = diff.max()
        elif dy > 0:
            diff[:dy, :] = diff.max()
        if dx < 0:
            diff[:, w + dx:] = diff.max()
        elif dx > 0:
            diff[:, :dx] = diff.max()
        costs.append(diff)
    return torch.stack(costs, dim=-1)


def _dijkstra_grid(edge_cost, source, offsets=None):
    offsets = offsets or _OFFSETS_8
    h, w, _ = edge_cost.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    sy, sx = int(source[0]), int(source[1])
    sy = min(max(sy, 0), h - 1)
    sx = min(max(sx, 0), w - 1)
    dist[sy, sx] = 0.0
    heap = [(0.0, sy, sx)]
    while heap:
        cur_dist, y, x = heappop(heap)
        if cur_dist > dist[y, x]:
            continue
        for k, (dy, dx) in enumerate(offsets):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            new_dist = cur_dist + float(edge_cost[y, x, k])
            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                heappush(heap, (new_dist, ny, nx))
    return dist


def _run_dijkstra(edge_cost, sources, neighborhood=8):
    offsets = _get_offsets(neighborhood)
    if neighborhood != 8 or official_dijkstra_image is None:
        outs = [_dijkstra_grid(edge_cost, src.tolist(), offsets=offsets) for src in sources]
        return np.stack(outs, axis=0)
    return official_dijkstra_image(
        np.ascontiguousarray(edge_cost.astype(np.float32)),
        np.ascontiguousarray(sources.astype(np.int32)),
    )


def _euclidean_distance_maps(height, width, sources, neighborhood=8):
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing='ij')
    outs = []
    diag_scale = float(np.sqrt(2.0)) if neighborhood == 8 else 1.0
    for source in sources:
        sy, sx = int(source[0]), int(source[1])
        dy = yy - float(sy)
        dx = xx - float(sx)
        if neighborhood == 4:
            dist = np.abs(dy) + np.abs(dx)
        else:
            dist = np.sqrt(dy * dy + dx * dx) / diag_scale
        outs.append(dist.astype(np.float32))
    return np.stack(outs, axis=0)


def _compute_supplier_centroids(assign_map, fallback_coords):
    num_suppliers, h, w = assign_map.shape
    centroids = []
    for idx in range(num_suppliers):
        mask = assign_map[idx] > 0
        if mask.any():
            ys, xs = torch.nonzero(mask, as_tuple=True)
            cy = int(torch.round(ys.float().mean()).item())
            cx = int(torch.round(xs.float().mean()).item())
            cy = min(max(cy, 0), h - 1)
            cx = min(max(cx, 0), w - 1)
            if not mask[cy, cx]:
                scores = (ys.float() - float(cy)).pow(2) + (xs.float() - float(cx)).pow(2)
                best_idx = int(torch.argmin(scores).item())
                cy = int(ys[best_idx].item())
                cx = int(xs[best_idx].item())
            centroids.append([cy, cx])
        else:
            centroids.append([int(fallback_coords[idx, 0].item()), int(fallback_coords[idx, 1].item())])
    return torch.tensor(centroids, device=assign_map.device, dtype=torch.long)


class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.05, max_iter=60):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, mu, nu, cost):
        kernel = torch.exp(-cost / self.eps).clamp_min(1e-8)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(self.max_iter):
            u = mu / (kernel @ v + 1e-8)
            v = nu / (kernel.transpose(0, 1) @ u + 1e-8)
        return u[:, None] * kernel * v[None, :]


class Point2MaskOT(nn.Module):
    def __init__(
        self,
        downsample=4,
        spatial_distance_mode="dijkstra",
        geodesic_neighborhood=8,
        path_cost_mode="learned",
        use_boundary_barrier=True,
        lambda_prob=1.0,
        lambda_boundary=0.4,
        lambda_feat=0.2,
        lambda_edge=0.0,
        sinkhorn_eps=0.05,
        sinkhorn_iters=60,
        use_low_level_edge=False,
        warmup_steps=1500,
        warmup_fg_ratio=0.08,
        min_fg_ratio=0.03,
        max_fg_ratio=0.30,
        semantic_ramp_steps=2500,
        enable_v2=True,
        fg_boundary_gamma=2.0,
        use_prediction_cost=True,
        pred_cost_weight=0.35,
        pred_warmup_steps=1000,
        pred_ramp_steps=2500,
        unary_semantic_weight=0.75,
        unary_boundary_weight=0.40,
        pseudo_mask_blend_weight=0.50,
        use_bg_exclusion=False,
        bg_exclusion_weight=0.75,
        bg_exclusion_tau=0.20,
        use_sam_prior=False,
        sam_prior_weight=0.20,
        sam_prior_warmup_steps=1500,
        sam_prior_ramp_steps=3000,
        sam_prior_mass_weight=0.25,
        use_synfoc_correction=False,
        synfoc_weight=0.20,
        synfoc_sam_teacher_steps=3000,
        synfoc_net_correction_warmup_steps=1500,
        synfoc_net_correction_ramp_steps=3000,
        synfoc_mass_weight=0.25,
    ):
        super().__init__()
        self.downsample = downsample
        self.spatial_distance_mode = spatial_distance_mode
        self.geodesic_neighborhood = geodesic_neighborhood
        self.path_cost_mode = path_cost_mode
        self.use_boundary_barrier = use_boundary_barrier
        self.lambda_prob = lambda_prob
        self.lambda_boundary = lambda_boundary
        self.lambda_feat = lambda_feat
        self.lambda_edge = lambda_edge
        self.use_low_level_edge = use_low_level_edge
        self.warmup_steps = warmup_steps
        self.warmup_fg_ratio = warmup_fg_ratio
        self.min_fg_ratio = min_fg_ratio
        self.max_fg_ratio = max_fg_ratio
        self.semantic_ramp_steps = semantic_ramp_steps
        self.enable_v2 = enable_v2
        self.fg_boundary_gamma = fg_boundary_gamma
        self.use_prediction_cost = use_prediction_cost
        self.pred_cost_weight = pred_cost_weight
        self.pred_warmup_steps = pred_warmup_steps
        self.pred_ramp_steps = pred_ramp_steps
        self.unary_semantic_weight = unary_semantic_weight
        self.unary_boundary_weight = unary_boundary_weight
        self.pseudo_mask_blend_weight = pseudo_mask_blend_weight
        self.use_bg_exclusion = use_bg_exclusion
        self.bg_exclusion_weight = bg_exclusion_weight
        self.bg_exclusion_tau = bg_exclusion_tau
        self.use_sam_prior = use_sam_prior
        self.sam_prior_weight = sam_prior_weight
        self.sam_prior_warmup_steps = sam_prior_warmup_steps
        self.sam_prior_ramp_steps = sam_prior_ramp_steps
        self.sam_prior_mass_weight = sam_prior_mass_weight
        self.use_synfoc_correction = use_synfoc_correction
        self.synfoc_weight = synfoc_weight
        self.synfoc_sam_teacher_steps = synfoc_sam_teacher_steps
        self.synfoc_net_correction_warmup_steps = synfoc_net_correction_warmup_steps
        self.synfoc_net_correction_ramp_steps = synfoc_net_correction_ramp_steps
        self.synfoc_mass_weight = synfoc_mass_weight
        self.sinkhorn = SinkhornDistance(eps=sinkhorn_eps, max_iter=sinkhorn_iters)
        self.sinkhorn_iters = sinkhorn_iters
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    def advance_step(self):
        self._step += 1

    def _semantic_blend_weight(self):
        if self.semantic_ramp_steps <= 0:
            return 1.0
        return min(float(self._step.item()) / float(self.semantic_ramp_steps), 1.0)

    def _target_fg_ratio(self, coarse_fg_score):
        if self._step.item() < self.warmup_steps:
            return float(self.warmup_fg_ratio)
        score_ratio = float(coarse_fg_score.mean().item())
        if self.enable_v2:
            support_ratio = float((coarse_fg_score > 0.60).float().mean().item())
            score_ratio = 0.5 * score_ratio + 0.5 * support_ratio
        return max(self.min_fg_ratio, min(self.max_fg_ratio, score_ratio))

    def _prediction_cost_weight(self):
        if not self.use_prediction_cost:
            return 0.0
        if self._step.item() < self.pred_warmup_steps:
            return 0.0
        if self.pred_ramp_steps <= 0:
            return self.pred_cost_weight
        ramp_step = min(self._step.item() - self.pred_warmup_steps, self.pred_ramp_steps)
        ramp = float(ramp_step) / float(self.pred_ramp_steps)
        return self.pred_cost_weight * (ramp ** 2)

    def _build_path_edge_cost(self, sem_ds, feat_ds, boundary_ds, edge_ds):
        offsets = _get_offsets(self.geodesic_neighborhood)
        if self.path_cost_mode == "uniform":
            step_lengths = _offset_step_lengths(offsets, device=sem_ds.device, dtype=sem_ds.dtype)
            return torch.ones(
                (*sem_ds.shape[-2:], len(offsets)),
                device=sem_ds.device,
                dtype=sem_ds.dtype,
            ) * step_lengths.view(1, 1, -1)
        if self.path_cost_mode != "learned":
            raise ValueError(f"Unsupported path_cost_mode: {self.path_cost_mode}")

        diff_prob = _neighbour_feature_cost(sem_ds, offsets=offsets)
        diff_feat = _neighbour_feature_cost(feat_ds, offsets=offsets)
        diff_all = self.lambda_prob * diff_prob + self.lambda_feat * diff_feat
        if self.use_boundary_barrier:
            diff_boundary = _neighbour_scalar_cost(boundary_ds, mode="max", offsets=offsets)
            diff_all = diff_all + self.lambda_boundary * diff_boundary
        if edge_ds is not None:
            diff_all = diff_all + self.lambda_edge * _neighbour_scalar_cost(edge_ds, mode="max", offsets=offsets)
        step_lengths = _offset_step_lengths(offsets, device=diff_all.device, dtype=diff_all.dtype)
        diff_all = diff_all * step_lengths.view(1, 1, -1)
        return diff_all.clamp_min(1e-4)

    def _sam_prior_cost_weight(self):
        if not self.use_sam_prior:
            return 0.0
        if self._step.item() < self.sam_prior_warmup_steps:
            return 0.0
        if self.sam_prior_ramp_steps <= 0:
            return float(self.sam_prior_weight)
        ramp_step = min(self._step.item() - self.sam_prior_warmup_steps, self.sam_prior_ramp_steps)
        ramp = float(ramp_step) / float(self.sam_prior_ramp_steps)
        return float(self.sam_prior_weight) * (ramp ** 2)

    def _synfoc_correction_weights(self):
        if not self.use_synfoc_correction:
            return 0.0, 0.0
        step = int(self._step.item())
        if self.synfoc_sam_teacher_steps <= 0:
            sam_teacher = 0.0
        else:
            sam_teacher = max(0.0, 1.0 - float(step) / float(self.synfoc_sam_teacher_steps))
        if step < self.synfoc_net_correction_warmup_steps:
            net_corrector = 0.0
        elif self.synfoc_net_correction_ramp_steps <= 0:
            net_corrector = 1.0
        else:
            ramp_step = min(step - self.synfoc_net_correction_warmup_steps, self.synfoc_net_correction_ramp_steps)
            net_corrector = float(ramp_step) / float(self.synfoc_net_correction_ramp_steps)
        return sam_teacher, net_corrector

    @torch.no_grad()
    def forward(self, image, feature, semantic_logits, boundary_logits, mask_logits, point_coords, point_labels, point_maps, sam_prior=None):
        batch_transport, batch_pseudo, batch_stats = [], [], []
        sem_prob = torch.softmax(semantic_logits, dim=1)
        boundary_prob = torch.sigmoid(boundary_logits)
        mask_prob = torch.sigmoid(mask_logits)
        edge_prior = _compute_image_edge(image) if self.use_low_level_edge else None
        if sam_prior is not None and sam_prior.dim() == 3:
            sam_prior = sam_prior.unsqueeze(1)

        for b in range(image.shape[0]):
            transport, pseudo, stats = self._single_image_ot(
                image=image[b],
                feature=feature[b],
                sem_prob=sem_prob[b],
                boundary_prob=boundary_prob[b, 0],
                mask_prob=mask_prob[b, 0],
                point_coords=point_coords[b],
                point_labels=point_labels[b],
                point_maps=point_maps[b],
                edge_prior=None if edge_prior is None else edge_prior[b, 0],
                sam_prior=None if sam_prior is None else sam_prior[b, 0],
            )
            batch_transport.append(transport)
            batch_pseudo.append(pseudo)
            batch_stats.append(stats)

        return (
            torch.stack(batch_transport, dim=0),
            torch.stack(batch_pseudo, dim=0),
            {
                "pseudo_fg_ratio": torch.stack([s["pseudo_fg_ratio"] for s in batch_stats]).mean(),
                "semantic_weight": torch.stack([s["semantic_weight"] for s in batch_stats]).mean(),
                "target_fg_ratio": torch.stack([s["target_fg_ratio"] for s in batch_stats]).mean(),
                "pred_cost_weight": torch.stack([s["pred_cost_weight"] for s in batch_stats]).mean(),
                "bg_exclusion_mean": torch.stack([s["bg_exclusion_mean"] for s in batch_stats]).mean(),
                "sam_prior_weight": torch.stack([s["sam_prior_weight"] for s in batch_stats]).mean(),
                "sam_prior_mean": torch.stack([s["sam_prior_mean"] for s in batch_stats]).mean(),
                "synfoc_sam_teacher": torch.stack([s["synfoc_sam_teacher"] for s in batch_stats]).mean(),
                "synfoc_net_corrector": torch.stack([s["synfoc_net_corrector"] for s in batch_stats]).mean(),
                "synfoc_conflict": torch.stack([s["synfoc_conflict"] for s in batch_stats]).mean(),
            },
        )

    def _single_image_ot(self, image, feature, sem_prob, boundary_prob, mask_prob, point_coords, point_labels, point_maps, edge_prior, sam_prior=None):
        h, w = sem_prob.shape[-2:]
        ds_h = max(8, h // self.downsample)
        ds_w = max(8, w // self.downsample)

        sem_ds = F.interpolate(sem_prob.unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0]
        feat_ds = F.interpolate(feature.unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0]
        boundary_ds = F.interpolate(boundary_prob.unsqueeze(0).unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0, 0]
        mask_ds = F.interpolate(mask_prob.unsqueeze(0).unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0, 0]
        edge_ds = None
        if edge_prior is not None:
            edge_ds = F.interpolate(edge_prior.unsqueeze(0).unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0, 0]
        sam_weight = self._sam_prior_cost_weight()
        sam_prior_ds = None
        if sam_prior is not None and (sam_weight > 0.0 or self.use_synfoc_correction):
            sam_prior_ds = F.interpolate(sam_prior.unsqueeze(0).unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False)[0, 0]
            sam_prior_ds = torch.nan_to_num(sam_prior_ds, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
        sam_prior_mean = sam_prior_ds.mean() if sam_prior_ds is not None else sem_prob.new_tensor(0.0)
        synfoc_sam_teacher, synfoc_net_corrector = self._synfoc_correction_weights()
        synfoc_fg_evidence = None
        synfoc_bg_evidence = None
        synfoc_conflict = sem_prob.new_tensor(0.0)
        if sam_prior_ds is not None and self.use_synfoc_correction:
            net_fg = mask_ds.clamp(1e-4, 1.0 - 1e-4)
            sam_fg = sam_prior_ds
            net_bg = 1.0 - net_fg
            sam_bg = 1.0 - sam_fg
            consensus_fg = net_fg * sam_fg
            consensus_bg = net_bg * sam_bg
            sam_only_fg = net_bg * sam_fg
            net_only_fg = net_fg * sam_bg
            sam_only_bg = net_fg * sam_bg
            net_only_bg = net_bg * sam_fg
            synfoc_fg_evidence = (
                consensus_fg
                + synfoc_sam_teacher * sam_only_fg
                + synfoc_net_corrector * net_only_fg
            ).clamp(1e-4, 1.0 - 1e-4)
            synfoc_bg_evidence = (
                consensus_bg
                + synfoc_sam_teacher * sam_only_bg
                + synfoc_net_corrector * net_only_bg
            ).clamp(1e-4, 1.0 - 1e-4)
            synfoc_conflict = (sam_only_fg + net_only_fg).mean()

        diff_all = self._build_path_edge_cost(sem_ds, feat_ds, boundary_ds, edge_ds)

        valid = point_labels >= 0
        coords = point_coords[valid]
        labels = point_labels[valid]
        if coords.numel() == 0:
            empty_transport = torch.stack([1.0 - sem_prob[1], sem_prob[1]], dim=0)
            empty_pseudo = (empty_transport[1:2] > empty_transport[0:1]).float()
            stats = {
                "pseudo_fg_ratio": sem_prob.new_tensor(float(empty_pseudo.mean().item())),
                "semantic_weight": sem_prob.new_tensor(self._semantic_blend_weight()),
                "target_fg_ratio": sem_prob.new_tensor(self.warmup_fg_ratio),
                "pred_cost_weight": sem_prob.new_tensor(self._prediction_cost_weight()),
                "bg_exclusion_mean": sem_prob.new_tensor(0.0),
                "sam_prior_weight": sem_prob.new_tensor(sam_weight),
                "sam_prior_mean": sam_prior_mean.detach(),
                "synfoc_sam_teacher": sem_prob.new_tensor(synfoc_sam_teacher),
                "synfoc_net_corrector": sem_prob.new_tensor(synfoc_net_corrector),
                "synfoc_conflict": synfoc_conflict.detach(),
                "sinkhorn_time_sec": sem_prob.new_tensor(0.0),
                "sinkhorn_time_ms": sem_prob.new_tensor(0.0),
                "sinkhorn_iters": sem_prob.new_tensor(float(self.sinkhorn_iters)),
            }
            return empty_transport, empty_pseudo, stats

        coords_ds = coords.clone().float()
        coords_ds[:, 0] = coords_ds[:, 0] * float(ds_h) / float(h)
        coords_ds[:, 1] = coords_ds[:, 1] * float(ds_w) / float(w)
        coords_ds = coords_ds.long()
        coords_ds[:, 0].clamp_(0, ds_h - 1)
        coords_ds[:, 1].clamp_(0, ds_w - 1)

        diff_np = diff_all.detach().cpu().numpy()
        coords_ds_np = coords_ds.detach().cpu().numpy()
        if self.spatial_distance_mode == "euclidean":
            dist_maps_np = _euclidean_distance_maps(ds_h, ds_w, coords_ds_np, neighborhood=self.geodesic_neighborhood)
        else:
            dist_maps_np = _run_dijkstra(diff_np, coords_ds_np, neighborhood=self.geodesic_neighborhood)
        dist_maps = torch.from_numpy(dist_maps_np).to(sem_prob.device)
        dist_maps = dist_maps / (dist_maps.amax(dim=(1, 2), keepdim=True) + 1e-6)

        fg_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
        bg_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
        fg_min_dist = dist_maps[fg_idx].min(dim=0)[0] if len(fg_idx) > 0 else torch.ones((ds_h, ds_w), device=dist_maps.device)
        bg_min_dist = dist_maps[bg_idx].min(dim=0)[0] if len(bg_idx) > 0 else torch.ones((ds_h, ds_w), device=dist_maps.device)
        if self.use_bg_exclusion and len(bg_idx) > 0:
            tau = max(float(self.bg_exclusion_tau), 1e-4)
            bg_exclusion_map = torch.exp(-bg_min_dist / tau).clamp(0.0, 1.0)
        else:
            bg_exclusion_map = torch.zeros((ds_h, ds_w), device=dist_maps.device)

        coarse_fg_score = torch.sigmoid((bg_min_dist - fg_min_dist) * 6.0).clamp(1e-4, 1.0 - 1e-4)
        coarse_bg_score = 1.0 - coarse_fg_score
        pred_weight = self._prediction_cost_weight()
        if pred_weight > 0.0:
            coarse_fg_score = ((1.0 - pred_weight) * coarse_fg_score + pred_weight * mask_ds).clamp(1e-4, 1.0 - 1e-4)
            coarse_bg_score = 1.0 - coarse_fg_score

        combined_boundary = boundary_ds
        if edge_ds is not None:
            combined_boundary = torch.maximum(combined_boundary, edge_ds)
        semantic_weight = self._semantic_blend_weight()
        class_prob = sem_ds[labels.long(), :, :]
        class_prior = torch.where(labels[:, None, None] == 1, coarse_fg_score.unsqueeze(0), coarse_bg_score.unsqueeze(0))
        class_prob = semantic_weight * class_prob + (1.0 - semantic_weight) * class_prior
        if self.enable_v2:
            fg_gate = (1.0 - combined_boundary).clamp_min(1e-4).pow(self.fg_boundary_gamma)
            class_prob = torch.where(labels[:, None, None] == 1, class_prob * fg_gate.unsqueeze(0), class_prob)
            coarse_fg_score = (coarse_fg_score * fg_gate).clamp(1e-4, 1.0 - 1e-4)
            coarse_bg_score = 1.0 - coarse_fg_score

        raw_dist_maps = dist_maps
        raw_dist_likeli = (1.0 - raw_dist_maps).clamp_min(0.0)
        init_likeli = raw_dist_likeli * class_prob
        init_assign = init_likeli.argmax(dim=0)
        init_assign_oh = F.one_hot(init_assign, coords_ds.shape[0]).permute(2, 0, 1).float()

        target_fg_ratio = self._target_fg_ratio(coarse_fg_score)
        if sam_prior_ds is not None and self.sam_prior_mass_weight > 0.0:
            mass_weight = min(max(float(self.sam_prior_mass_weight), 0.0), 1.0)
            prior_ratio = sam_prior_mean.clamp(self.min_fg_ratio, self.max_fg_ratio)
            target_fg_ratio = ((1.0 - mass_weight) * target_fg_ratio + mass_weight * prior_ratio).clamp(self.min_fg_ratio, self.max_fg_ratio)
        if synfoc_fg_evidence is not None and self.synfoc_mass_weight > 0.0:
            mass_weight = min(max(float(self.synfoc_mass_weight), 0.0), 1.0)
            evidence_ratio = synfoc_fg_evidence.mean().clamp(self.min_fg_ratio, self.max_fg_ratio)
            target_fg_ratio = ((1.0 - mass_weight) * target_fg_ratio + mass_weight * evidence_ratio).clamp(self.min_fg_ratio, self.max_fg_ratio)
        if self._step.item() <= self.warmup_steps:
            supplier_mass = init_assign_oh.sum(dim=(1, 2)).float().clamp_min(1.0)
            supplier_pseudo_source = init_assign_oh
        else:
            centroid_coords = _compute_supplier_centroids(init_assign_oh, coords_ds)
            centroid_coords_np = centroid_coords.detach().cpu().numpy()
            if self.spatial_distance_mode == "euclidean":
                refined_dist_maps_np = _euclidean_distance_maps(ds_h, ds_w, centroid_coords_np, neighborhood=self.geodesic_neighborhood)
            else:
                refined_dist_maps_np = _run_dijkstra(diff_np, centroid_coords_np, neighborhood=self.geodesic_neighborhood)
            refined_dist_maps = torch.from_numpy(refined_dist_maps_np).to(sem_prob.device)
            refined_dist_maps = refined_dist_maps / (refined_dist_maps.amax(dim=(1, 2), keepdim=True) + 1e-6)
            refined_dist_likeli = (1.0 - refined_dist_maps).clamp_min(0.0)
            refined_likeli = refined_dist_likeli * class_prob
            refined_assign = refined_likeli.argmax(dim=0)
            supplier_pseudo_source = F.one_hot(refined_assign, coords_ds.shape[0]).permute(2, 0, 1).float()
            supplier_mass = supplier_pseudo_source.sum(dim=(1, 2)).float().clamp_min(1.0)

        supplier_mass = supplier_mass / supplier_mass.sum().clamp_min(1e-6)
        if len(fg_idx) > 0:
            fg_mass = supplier_mass[fg_idx]
            fg_mass = fg_mass / fg_mass.sum().clamp_min(1e-6) * target_fg_ratio
            supplier_mass[fg_idx] = fg_mass
        if len(bg_idx) > 0:
            bg_mass = supplier_mass[bg_idx]
            bg_mass = bg_mass / bg_mass.sum().clamp_min(1e-6) * (1.0 - target_fg_ratio)
            supplier_mass[bg_idx] = bg_mass
        supplier_mass = supplier_mass / supplier_mass.sum().clamp_min(1e-6)

        semantic_unary = (-torch.log(class_prob.clamp_min(1e-4))).reshape(coords_ds.shape[0], -1)
        boundary_unary = torch.where(labels[:, None, None] == 1, combined_boundary.unsqueeze(0), torch.zeros_like(class_prob)).reshape(coords_ds.shape[0], -1)
        mask_unary_map = torch.where(
            labels[:, None, None] == 1,
            -torch.log(mask_ds.clamp(1e-4, 1.0 - 1e-4)).unsqueeze(0),
            -torch.log((1.0 - mask_ds).clamp(1e-4, 1.0 - 1e-4)).unsqueeze(0),
        ).reshape(coords_ds.shape[0], -1)
        if sam_prior_ds is not None:
            sam_prior_unary = torch.where(
                labels[:, None, None] == 1,
                -torch.log(sam_prior_ds).unsqueeze(0),
                -torch.log(1.0 - sam_prior_ds).unsqueeze(0),
            ).reshape(coords_ds.shape[0], -1)
        else:
            sam_prior_unary = torch.zeros_like(mask_unary_map)
        if synfoc_fg_evidence is not None and synfoc_bg_evidence is not None:
            synfoc_unary = torch.where(
                labels[:, None, None] == 1,
                -torch.log(synfoc_fg_evidence).unsqueeze(0),
                -torch.log(synfoc_bg_evidence).unsqueeze(0),
            ).reshape(coords_ds.shape[0], -1)
        else:
            synfoc_unary = torch.zeros_like(mask_unary_map)
        bg_exclusion_unary = torch.where(
            labels[:, None, None] == 1,
            bg_exclusion_map.unsqueeze(0),
            torch.zeros_like(class_prob),
        ).reshape(coords_ds.shape[0], -1)

        cost = raw_dist_maps.reshape(raw_dist_maps.shape[0], -1)
        cost = cost + self.unary_semantic_weight * semantic_unary + self.unary_boundary_weight * boundary_unary + pred_weight * mask_unary_map + sam_weight * sam_prior_unary + self.synfoc_weight * synfoc_unary + self.bg_exclusion_weight * bg_exclusion_unary
        cost = torch.nan_to_num(cost, nan=10.0, posinf=10.0, neginf=0.0)
        cost = cost - cost.amin(dim=1, keepdim=True)
        cost = cost.clamp(0.0, 10.0)

        consumer_mass = torch.full((cost.shape[1],), 1.0 / float(cost.shape[1]), device=cost.device, dtype=cost.dtype)
        if cost.is_cuda:
            torch.cuda.synchronize(cost.device)
        sinkhorn_start = time.perf_counter()
        plan = self.sinkhorn(supplier_mass, consumer_mass, cost)
        if cost.is_cuda:
            torch.cuda.synchronize(cost.device)
        sinkhorn_time_sec = time.perf_counter() - sinkhorn_start
        plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0).view(coords_ds.shape[0], ds_h, ds_w)
        supplier_assign = plan.argmax(dim=0)
        supplier_pseudo = F.one_hot(supplier_assign, coords_ds.shape[0]).permute(2, 0, 1).float()

        fg_transport = plan[fg_idx].sum(dim=0) if len(fg_idx) > 0 else torch.zeros((ds_h, ds_w), device=plan.device)
        bg_transport = plan[bg_idx].sum(dim=0) if len(bg_idx) > 0 else torch.zeros((ds_h, ds_w), device=plan.device)
        transport = torch.stack([bg_transport, fg_transport], dim=0).unsqueeze(0)
        transport = F.interpolate(transport, size=(h, w), mode="bilinear", align_corners=False)[0]
        transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)
        transport = transport / (transport.sum(dim=0, keepdim=True) + 1e-6)

        transport[1][point_maps[0] > 0] = 1.0
        transport[0][point_maps[0] > 0] = 0.0
        transport[0][point_maps[1] > 0] = 1.0
        transport[1][point_maps[1] > 0] = 0.0

        fg_score_ds = supplier_pseudo[fg_idx].sum(dim=0) if len(fg_idx) > 0 else torch.zeros((ds_h, ds_w), device=plan.device)
        fg_score = F.interpolate(fg_score_ds.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        if pred_weight > 0.0:
            fg_score = ((1.0 - self.pseudo_mask_blend_weight * pred_weight) * fg_score + (self.pseudo_mask_blend_weight * pred_weight) * mask_prob).clamp(0.0, 1.0)
        if self.enable_v2:
            full_boundary = boundary_prob
            if edge_prior is not None:
                full_boundary = torch.maximum(full_boundary, edge_prior)
            full_fg_gate = (1.0 - full_boundary).clamp_min(1e-4).pow(self.fg_boundary_gamma)
            fg_score = (fg_score * full_fg_gate).clamp(0.0, 1.0)
        pseudo_hard = _build_connected_area_mask(fg_score, target_fg_ratio, fg_point_mask=point_maps[0], bg_point_mask=point_maps[1])
        pseudo = pseudo_hard.unsqueeze(0)
        target_fg_ratio_tensor = target_fg_ratio if torch.is_tensor(target_fg_ratio) else transport.new_tensor(target_fg_ratio)
        stats = {
            "pseudo_fg_ratio": transport.new_tensor(float(pseudo.mean().item())),
            "semantic_weight": transport.new_tensor(semantic_weight),
            "target_fg_ratio": target_fg_ratio_tensor,
            "pred_cost_weight": transport.new_tensor(pred_weight),
            "bg_exclusion_mean": transport.new_tensor(float(bg_exclusion_map.mean().item())),
            "sam_prior_weight": transport.new_tensor(sam_weight),
            "sam_prior_mean": transport.new_tensor(float(sam_prior_mean.item())),
            "synfoc_sam_teacher": transport.new_tensor(synfoc_sam_teacher),
            "synfoc_net_corrector": transport.new_tensor(synfoc_net_corrector),
            "synfoc_conflict": transport.new_tensor(float(synfoc_conflict.item())),
            "sinkhorn_time_sec": transport.new_tensor(float(sinkhorn_time_sec)),
            "sinkhorn_time_ms": transport.new_tensor(float(sinkhorn_time_sec * 1000.0)),
            "sinkhorn_iters": transport.new_tensor(float(self.sinkhorn_iters)),
        }
        return transport, pseudo, stats
