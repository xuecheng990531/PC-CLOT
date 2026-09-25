"""Optimal Transport generator — produces per-pixel transport probability from
point_maps + network predictions.

Interface:
    forward(point_maps, feature, mask_logits, boundary_logits) → T ∈ [B, 2, H, W]
    T[:, 0] = background probability, T[:, 1] = foreground probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_cost(cost, eps=1e-6):
    """Min-max normalize to [0, 1] per batch item."""
    B = cost.shape[0]
    c_flat = cost.reshape(B, -1)
    c_min = c_flat.min(dim=-1, keepdim=True)[0].reshape(B, *([1] * (cost.ndim - 1)))
    c_max = c_flat.max(dim=-1, keepdim=True)[0].reshape(B, *([1] * (cost.ndim - 1)))
    return (cost - c_min) / (c_max - c_min + eps)


def _compute_point_distance(point_map, eps=1e-6):
    """Euclidean distance from each pixel to the nearest point (raw pixels)."""
    H, W = point_map.shape
    device = point_map.device

    mask = point_map > 0
    if mask.sum() == 0:
        return torch.full((H, W), float(H + W), device=device)

    ys, xs = torch.where(mask)
    points = torch.stack([ys.float(), xs.float()], dim=-1)

    y_grid = torch.arange(H, device=device).float()
    x_grid = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    grid = torch.stack([yy, xx], dim=-1)

    diff = grid.unsqueeze(2) - points.view(1, 1, -1, 2)
    dist = torch.sqrt((diff ** 2).sum(dim=3) + eps)
    return dist.min(dim=2).values


class OTGenerator(nn.Module):
    """Point2Mask-style Optimal Transport pseudo-mask generator.

    Cost components:
        pred  — mask prediction cost: -log(sigmoid) for fg, -log(1-sigmoid) for bg
        dist  — Euclidean distance to nearest fg/bg point
        feat  — cosine distance to fg/bg seed features
        bdry  — boundary penalty

    Output is per-pixel normalized so T_fg + T_bg = 1 per pixel.
    """

    def __init__(
        self,
        num_iters=50,
        eps=1e-6,
        temperature=1.0,
        cost_weights=None,
        min_fg_ratio=0.01,
        max_fg_ratio=0.30,
        use_soft_transport=True,
        warmup_steps=1000,
        ramp_steps=2000,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
        self.temperature = temperature
        self.min_fg_ratio = min_fg_ratio
        self.max_fg_ratio = max_fg_ratio
        self.use_soft_transport = use_soft_transport
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps

        self.cost_weights = {'pred': 1.0, 'dist': 1.0, 'feat': 1.0, 'bdry': 0.2}
        if cost_weights is not None:
            self.cost_weights.update(cost_weights)

        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    def advance_step(self):
        self._step += 1

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------

    def _prediction_cost(self, mask_logits):
        prob = torch.sigmoid(mask_logits)
        C_fg = -torch.log(prob + self.eps).squeeze(1)
        C_bg = -torch.log(1 - prob + self.eps).squeeze(1)
        return C_bg, C_fg

    def _distance_cost(self, point_maps):
        B, _, H, W = point_maps.shape
        C_dist_bg, C_dist_fg = [], []
        for b in range(B):
            C_dist_fg.append(_compute_point_distance(point_maps[b, 0], self.eps))
            C_dist_bg.append(_compute_point_distance(point_maps[b, 1], self.eps))
        return (torch.stack(C_dist_bg, dim=0),
                torch.stack(C_dist_fg, dim=0))

    def _feature_cost(self, feature, point_maps):
        B, C, H, W = feature.shape

        fg_weight = point_maps[:, 0:1]
        bg_weight = point_maps[:, 1:2]

        fg_sum = fg_weight.reshape(B, -1).sum(dim=1, keepdim=True)
        bg_sum = bg_weight.reshape(B, -1).sum(dim=1, keepdim=True)

        feat_flat = feature.reshape(B, C, -1)

        fg_seed = (feat_flat * fg_weight.reshape(B, 1, -1)).sum(dim=2) / (fg_sum + self.eps)
        bg_seed = (feat_flat * bg_weight.reshape(B, 1, -1)).sum(dim=2) / (bg_sum + self.eps)

        feat_norm = F.normalize(feat_flat, p=2, dim=1)
        fg_seed_norm = F.normalize(fg_seed, p=2, dim=1)
        bg_seed_norm = F.normalize(bg_seed, p=2, dim=1)

        cos_fg = (feat_norm * fg_seed_norm.unsqueeze(2)).sum(dim=1)
        cos_bg = (feat_norm * bg_seed_norm.unsqueeze(2)).sum(dim=1)

        C_feat_fg = (1 - cos_fg).reshape(B, H, W)
        C_feat_bg = (1 - cos_bg).reshape(B, H, W)
        return C_feat_bg, C_feat_fg

    def _boundary_cost(self, boundary_logits):
        boundary_prob = torch.sigmoid(boundary_logits).squeeze(1)
        return boundary_prob, boundary_prob

    # ------------------------------------------------------------------
    # Mass
    # ------------------------------------------------------------------

    def _estimate_mass(self, mask_logits):
        B, _, H, W = mask_logits.shape

        step = int(self._step.item())

        if step < 1000:
            # Warm-up: conservative fixed foreground mass
            fg_ratio = torch.full((B,), 0.10, device=mask_logits.device)
        else:
            # Use network prediction, detach to avoid gradient through mass
            mask_prob = torch.sigmoid(mask_logits.detach())
            fg_ratio = mask_prob.reshape(B, -1).mean(dim=1)
            fg_ratio = torch.clamp(fg_ratio, self.min_fg_ratio, self.max_fg_ratio)

        return torch.stack([1.0 - fg_ratio, fg_ratio], dim=1)

    # ------------------------------------------------------------------
    # Sinkhorn
    # ------------------------------------------------------------------

    def _sinkhorn(self, cost, a, b):
        B, K, N = cost.shape
        device = cost.device

        K_mat = torch.exp(-cost / self.temperature)
        u = torch.ones(B, K, device=device)
        v = torch.ones(B, N, device=device)

        for _ in range(self.num_iters):
            u = a / (torch.bmm(K_mat, v.unsqueeze(2)).squeeze(2) + self.eps)
            v = b / (torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + self.eps)

        return u.unsqueeze(2) * K_mat * v.unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, point_maps, feature, mask_logits, boundary_logits):
        """
        Args:
            point_maps:     [B, 2, H, W]  ch0=fg, ch1=bg
            feature:        [B, C, H, W]
            mask_logits:    [B, 1, H, W]
            boundary_logits:[B, 1, H, W]

        Returns:
            T: [B, 2, H, W]  per-pixel class probability
               T[:, 0] = bg prob, T[:, 1] = fg prob, T_fg + T_bg = 1
        """
        B, _, H, W = point_maps.shape
        N = H * W

        # Step-dependent cost weights: early on, trust only distance
        step = int(self._step.item())
        if step < self.warmup_steps:
            w_eff = {'pred': 0.0, 'dist': 1.0, 'feat': 0.1, 'bdry': 0.0}
        elif step < self.warmup_steps + self.ramp_steps:
            alpha = (step - self.warmup_steps) / self.ramp_steps
            w_eff = {
                'pred': self.cost_weights['pred'] * alpha,
                'dist': self.cost_weights['dist'],
                'feat': self.cost_weights['feat'] * (0.1 + 0.9 * alpha),
                'bdry': 0.0,  # keep off until geodesic cost is implemented
            }
        else:
            w_eff = {k: v for k, v in self.cost_weights.items()}
            w_eff['bdry'] = 0.0  # boundary cost currently non-discriminative

        # Cost components
        C_pred_bg, C_pred_fg = self._prediction_cost(mask_logits)
        C_dist_bg, C_dist_fg = self._distance_cost(point_maps)
        C_feat_bg, C_feat_fg = self._feature_cost(feature, point_maps)
        C_bdry_bg, C_bdry_fg = self._boundary_cost(boundary_logits)

        cost_bg = (w_eff['pred'] * C_pred_bg + w_eff['dist'] * C_dist_bg +
                   w_eff['feat'] * C_feat_bg + w_eff['bdry'] * C_bdry_bg)
        cost_fg = (w_eff['pred'] * C_pred_fg + w_eff['dist'] * C_dist_fg +
                   w_eff['feat'] * C_feat_fg + w_eff['bdry'] * C_bdry_fg)

        cost = torch.stack([cost_bg, cost_fg], dim=1).reshape(B, 2, N)
        cost = _normalize_cost(cost, self.eps)

        # Mass
        a = self._estimate_mass(mask_logits)
        b = torch.ones(B, N, device=cost.device) / N

        # Sinkhorn
        T = self._sinkhorn(cost, a, b)

        # Per-pixel normalize
        T_prob = T.reshape(B, 2, H, W)
        T_sum = T_prob.sum(dim=1, keepdim=True).clamp(min=self.eps)
        T_prob = T_prob / T_sum

        # Assertion
        if torch.isnan(T_prob).any():
            raise RuntimeError("OT transport contains NaN")

        # Early training: use hard argmax target (suppresses false positives)
        if not self.use_soft_transport or step < self.warmup_steps + self.ramp_steps:
            hard_idx = torch.argmax(T_prob, dim=1, keepdim=True)
            T_hard = torch.zeros_like(T_prob)
            T_hard.scatter_(1, hard_idx, 1.0)
            return T_hard.float()

        return T_prob
