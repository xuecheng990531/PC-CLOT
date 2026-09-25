"""Prototype pooling for region-aware prototype aggregation."""

import torch
import torch.nn as nn


class PrototypePooling(nn.Module):
    """Pool feature maps into region prototypes.

    Input:
        feature:       [B, C, H, W]
        transport:     [B, 2, H, W]  per-pixel normalized (T_fg + T_bg = 1)
        boundary_logits: [B, 1, H, W]  optional boundary prior
        semantic_logits: [B, 2, H, W]  optional semantic logits for uncertainty
        mask_logits:     [B, 1, H, W]  optional mask logits for uncertainty

    Output:
        By default returns `prototypes: [B, K, C]`.
        If `return_dict=True`, returns a dict with named prototypes and weights.
    """

    def __init__(
        self,
        eps=1e-6,
        use_boundary_prototype=False,
        use_uncertainty_prototype=False,
        boundary_mix=0.5,
        uncertainty_sem_weight=0.5,
        uncertainty_mask_weight=0.5,
        return_dict=False,
    ):
        super().__init__()
        self.eps = eps
        self.use_boundary_prototype = use_boundary_prototype
        self.use_uncertainty_prototype = use_uncertainty_prototype
        self.boundary_mix = boundary_mix
        self.uncertainty_sem_weight = uncertainty_sem_weight
        self.uncertainty_mask_weight = uncertainty_mask_weight
        self.return_dict = return_dict

    def _pool_with_weight(self, feat_flat, weight_flat):
        weight_sum = weight_flat.sum(dim=2) + self.eps
        return (weight_flat * feat_flat).sum(dim=2) / weight_sum

    def _semantic_entropy(self, semantic_logits):
        if semantic_logits is None:
            return None
        prob = torch.softmax(semantic_logits, dim=1).clamp_min(self.eps)
        entropy = -(prob * prob.log()).sum(dim=1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(prob.shape[1], device=prob.device, dtype=prob.dtype))
        return entropy

    def _mask_uncertainty(self, mask_logits):
        if mask_logits is None:
            return None
        prob = torch.sigmoid(mask_logits)
        return 1.0 - torch.abs(2.0 * prob - 1.0)

    def forward(self, feature, transport, boundary_logits=None, semantic_logits=None, mask_logits=None):
        B, C, H, W = feature.shape

        feat_flat = feature.reshape(B, C, H * W)
        T_bg = transport[:, 0:1].reshape(B, 1, H * W)
        T_fg = transport[:, 1:2].reshape(B, 1, H * W)

        r_bg = self._pool_with_weight(feat_flat, T_bg)
        r_fg = self._pool_with_weight(feat_flat, T_fg)

        prototype_list = [r_bg, r_fg]
        prototype_names = ["bg", "fg"]
        weight_maps = {
            "bg": T_bg.reshape(B, 1, H, W),
            "fg": T_fg.reshape(B, 1, H, W),
        }

        if self.use_boundary_prototype:
            if boundary_logits is None:
                boundary_prob = 4.0 * transport[:, 0:1] * transport[:, 1:2]
            else:
                boundary_prob = torch.sigmoid(boundary_logits)
                transport_mix = 4.0 * transport[:, 0:1] * transport[:, 1:2]
                boundary_prob = self.boundary_mix * boundary_prob + (1.0 - self.boundary_mix) * transport_mix
            boundary_weight = boundary_prob.reshape(B, 1, H * W)
            r_boundary = self._pool_with_weight(feat_flat, boundary_weight)
            prototype_list.append(r_boundary)
            prototype_names.append("boundary")
            weight_maps["boundary"] = boundary_weight.reshape(B, 1, H, W)

        if self.use_uncertainty_prototype:
            semantic_uncertainty = self._semantic_entropy(semantic_logits)
            mask_uncertainty = self._mask_uncertainty(mask_logits)
            if semantic_uncertainty is None and mask_uncertainty is None:
                uncertainty_map = 4.0 * transport[:, 0:1] * transport[:, 1:2]
            else:
                uncertainty_terms = []
                active_weight_sum = 0.0
                if semantic_uncertainty is not None:
                    uncertainty_terms.append(self.uncertainty_sem_weight * semantic_uncertainty)
                    active_weight_sum += self.uncertainty_sem_weight
                if mask_uncertainty is not None:
                    uncertainty_terms.append(self.uncertainty_mask_weight * mask_uncertainty)
                    active_weight_sum += self.uncertainty_mask_weight
                uncertainty_map = torch.stack(uncertainty_terms, dim=0).sum(dim=0)
                uncertainty_map = uncertainty_map / max(active_weight_sum, self.eps)
            uncertainty_map = uncertainty_map.clamp(0.0, 1.0)
            uncertainty_weight = uncertainty_map.reshape(B, 1, H * W)
            r_uncertainty = self._pool_with_weight(feat_flat, uncertainty_weight)
            prototype_list.append(r_uncertainty)
            prototype_names.append("uncertainty")
            weight_maps["uncertainty"] = uncertainty_weight.reshape(B, 1, H, W)

        prototypes = torch.stack(prototype_list, dim=1)
        if not self.return_dict:
            return prototypes

        named = {name: prototype for name, prototype in zip(prototype_names, prototype_list)}
        return {
            "prototypes": prototypes,
            "prototype_names": prototype_names,
            "named_prototypes": named,
            "weight_maps": weight_maps,
        }
