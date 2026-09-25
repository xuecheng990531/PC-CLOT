"""Boundary affinity loss — supervise boundary via transport neighborhood consistency."""

import torch
import torch.nn.functional as F


def boundary_affinity_loss(boundary_logits, transport, eps=1e-6):
    """Boundary affinity loss from pseudo-mask neighborhood consistency.

    Args:
        boundary_logits: [B, 1, H, W]
        transport:       [B, 2, H, W]  per-pixel normalized

    Returns:
        scalar loss
    """
    B, _, H, W = transport.shape
    boundary_logits = torch.nan_to_num(boundary_logits, nan=0.0, posinf=20.0, neginf=-20.0)
    transport = torch.nan_to_num(transport, nan=0.0, posinf=1.0, neginf=0.0)
    boundary_prob = torch.sigmoid(boundary_logits).clamp(0.0, 1.0)

    pseudo_label = transport[:, 1] > transport[:, 0]  # [B, H, W]
    pseudo_label_flat = pseudo_label.unsqueeze(1).float()

    patches_label = F.unfold(pseudo_label_flat, kernel_size=3, padding=1)
    patches_bdry = F.unfold(boundary_prob, kernel_size=3, padding=1)

    center_label = patches_label[:, 4:5]
    center_bdry = patches_bdry[:, 4:5]

    neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    total_loss = 0.0
    total_pairs = 0

    for k in neighbor_indices:
        nbr_label = patches_label[:, k:k + 1]
        nbr_bdry = patches_bdry[:, k:k + 1]

        A_kl = 1.0 - torch.max(center_bdry, nbr_bdry)
        A_kl = torch.nan_to_num(A_kl, nan=0.5, posinf=1.0, neginf=0.0)
        A_kl = torch.clamp(A_kl, 0.0, 1.0)

        target_aff = (center_label == nbr_label).float()
        target_aff = torch.nan_to_num(target_aff, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        total_loss += F.binary_cross_entropy(A_kl, target_aff, reduction='sum')
        total_pairs += A_kl.numel()

    if total_pairs == 0:
        return torch.tensor(0.0, device=transport.device, requires_grad=True)

    return total_loss / total_pairs
