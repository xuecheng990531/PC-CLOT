import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_utils import get_groupnorm


class nmODEBlock(nn.Module):
    """nmODE-style update block: Fi = -Yi + f(Yi + g(Xi))."""

    def __init__(self, in_channels, memory_channels):
        super().__init__()
        self.projector_g = nn.Sequential(
            nn.Conv2d(in_channels, memory_channels, kernel_size=1),
            get_groupnorm(memory_channels),
            nn.ReLU(inplace=True)
        )
        self.function_f = nn.Sequential(
            nn.Conv2d(memory_channels, memory_channels, kernel_size=3, padding=1),
            get_groupnorm(memory_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(memory_channels, memory_channels, kernel_size=3, padding=1),
            get_groupnorm(memory_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, Xi, Yi):
        """
        Args:
            Xi: current skip feature [B, C_in, H, W]
            Yi: memory state [B, memory_channels, H, W]

        Returns:
            Fi: update [B, memory_channels, H, W]
        """
        g_Xi = self.projector_g(Xi)
        g_Xi = torch.nan_to_num(g_Xi, nan=0.0, posinf=1e4, neginf=-1e4)
        combined = Yi + g_Xi
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)
        f_combined = self.function_f(combined)
        f_combined = torch.nan_to_num(f_combined, nan=0.0, posinf=1e4, neginf=-1e4)
        Fi = -Yi + f_combined
        Fi = torch.nan_to_num(Fi, nan=0.0, posinf=1e4, neginf=-1e4)
        return Fi


class PCSkipFusion(nn.Module):
    """Predictor-Corrector enhanced skip fusion with per-forward local memory.

    Process order: deep to shallow [x4, x3, x2, x1]
    Each processed feature is enhanced via memory state.
    Memory Y and Fi history are only used inside the current forward pass.
    """

    def __init__(self, input_channels, memory_channels=32, delta=1.0,
                 residual_scale=0.1, state_clip=10.0):
        """
        Args:
            input_channels: list [64, 128, 256, 512] (shallow to deep)
            memory_channels: hidden dimension for memory state (default 32)
            delta: ODE step size (default 1.0)
        """
        super().__init__()
        self.input_channels = input_channels  # [64, 128, 256, 512]
        self.memory_channels = memory_channels
        self.delta = delta
        self.residual_scale = residual_scale
        self.state_clip = state_clip

        self.ode_blocks = nn.ModuleList([
            nmODEBlock(in_c, memory_channels) for in_c in input_channels
        ])

        self.proj_outs = nn.ModuleList([
            nn.Conv2d(memory_channels, in_c, kernel_size=1) for in_c in input_channels
        ])

    def _get_spatial_key(self, H, W):
        """Generate unique key for spatial dimensions."""
        return (H, W)

    def _initialize_memory(self, memory_bank, spatial_key, device):
        """Initialize memory state for a spatial scale if not present."""
        if spatial_key not in memory_bank:
            H, W = spatial_key
            memory_bank[spatial_key] = torch.zeros(
                1, self.memory_channels, H, W, device=device
            )

    def _resize_tensor(self, tensor, H, W):
        """Resize tensor to target spatial dimensions."""
        if tensor.shape[-2:] == (H, W):
            return tensor
        return F.interpolate(tensor, size=(H, W), mode='bilinear', align_corners=False)

    def _predictor_step(self, Y, Fi_hist_resized):
        """Adams-Bashforth predictor.

        Args:
            Y: current memory state [B, C, H, W]
            Fi_hist_resized: list of historical Fi, all resized to current spatial dims

        Returns:
            Y_pred: predicted next state
        """
        num_hist = len(Fi_hist_resized)

        if num_hist == 1:
            Y_pred = Y + self.delta * Fi_hist_resized[-1]
        elif num_hist == 2:
            Y_pred = Y + self.delta / 2 * (3 * Fi_hist_resized[-1] - Fi_hist_resized[-2])
        elif num_hist == 3:
            Y_pred = Y + self.delta / 12 * (23 * Fi_hist_resized[-1] - 16 * Fi_hist_resized[-2] + 5 * Fi_hist_resized[-3])
        else:  # num_hist >= 4
            Y_pred = Y + self.delta / 24 * (55 * Fi_hist_resized[-1] - 59 * Fi_hist_resized[-2] + 37 * Fi_hist_resized[-3] - 9 * Fi_hist_resized[-4])

        return Y_pred

    def _corrector_step(self, Y, F_pred, Fi_hist_resized):
        """Adams-Moulton corrector.

        Args:
            Y: current memory state [B, C, H, W]
            F_pred: predicted Fi from predictor
            Fi_hist_resized: list of historical Fi, all resized to current spatial dims

        Returns:
            Y_next: corrected next state
        """
        num_hist = len(Fi_hist_resized)

        if num_hist == 1:
            Y_next = Y + self.delta / 2 * (Fi_hist_resized[-1] + F_pred)
        elif num_hist == 2:
            Y_next = Y + self.delta / 12 * (5 * F_pred + 8 * Fi_hist_resized[-1] - Fi_hist_resized[-2])
        else:  # num_hist >= 3
            Y_next = Y + self.delta / 24 * (9 * F_pred + 19 * Fi_hist_resized[-1] - 5 * Fi_hist_resized[-2] + Fi_hist_resized[-3])

        return Y_next

    def _stabilize_state(self, tensor):
        """Keep predictor/corrector state in a bounded numeric range."""
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=self.state_clip, neginf=-self.state_clip)
        return tensor.clamp(min=-self.state_clip, max=self.state_clip)

    def forward(self, skips):
        """
        Args:
            skips: [x1, x2, x3, x4] (shallow to deep)

        Returns:
            enhanced_skips: [s1, s2, s3, s4] (same shape as inputs)
        """
        enhanced_skips = [None] * len(skips)
        process_order = [3, 2, 1, 0]
        memory_bank = {}
        fi_history = []

        for idx in process_order:
            Xi = skips[idx]
            B, C_in, H, W = Xi.shape
            spatial_key = self._get_spatial_key(H, W)
            device = Xi.device

            self._initialize_memory(memory_bank, spatial_key, device)

            Y = memory_bank[spatial_key]
            if Y.shape[0] != B:
                Y = Y.expand(B, -1, -1, -1)
            Y = self._stabilize_state(Y)

            ode_block = self.ode_blocks[idx]
            Fi = ode_block(Xi, Y)
            Fi = self._stabilize_state(Fi)

            Fi_hist_resized = []
            for F_hist in fi_history:
                F_hist_resized = self._resize_tensor(F_hist, H, W)
                Fi_hist_resized.append(F_hist_resized)

            if len(Fi_hist_resized) > 0:
                Y_pred = self._predictor_step(Y, Fi_hist_resized)
                Y_pred = self._stabilize_state(Y_pred)
                F_pred = ode_block(Xi, Y_pred)
                F_pred = self._stabilize_state(F_pred)
                Y_next = self._corrector_step(Y, F_pred, Fi_hist_resized)
            else:
                Y_next = Y + self.delta * Fi

            Y_next = self._stabilize_state(Y_next)
            memory_bank[spatial_key] = Y_next.detach()
            fi_history.append(Fi.detach())

            if len(fi_history) > 4:
                fi_history.pop(0)

            proj_out = self.proj_outs[idx]
            residual = torch.tanh(proj_out(Y_next))
            residual = self._stabilize_state(residual)
            enhanced_Xi = Xi + self.residual_scale * residual
            enhanced_Xi = self._stabilize_state(enhanced_Xi)

            enhanced_skips[idx] = enhanced_Xi

        return enhanced_skips

    def reset_memory(self):
        """No-op kept for compatibility with older call sites."""
        return None
