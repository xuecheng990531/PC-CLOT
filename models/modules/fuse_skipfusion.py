import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_utils import get_groupnorm


class FuseSkipProjector(nn.Module):
    """Project one skip feature to the shared full-resolution memory grid."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            get_groupnorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size):
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.proj(x)


class FuseUNetSkipFusion(nn.Module):
    """FuseUNet-style multi-scale skip fusion on a shared memory stream.

    This is a 2D adaptation for the V3 branch:
      - all skip features are projected to the shallowest resolution
      - a predictor-corrector memory update runs from deep to shallow
      - the final memory map is projected back to the head feature space
    """

    def __init__(
        self,
        skip_channels,
        memory_channels,
        out_channels,
        delta=0.1,
        state_clip=10.0,
        selected_indices=(1, 2, 4),
        target_index=1,
        pcsc_mode="full",
    ):
        super().__init__()
        if pcsc_mode not in {"full", "ab_predictor", "am_corrector"}:
            raise ValueError(f"Unsupported pcsc_mode: {pcsc_mode}")
        self.selected_indices = tuple(selected_indices)
        self.target_index = int(target_index)
        self.num_stages = len(self.selected_indices)
        self.memory_channels = memory_channels
        self.delta = delta
        self.state_clip = state_clip
        self.pcsc_mode = pcsc_mode
        self.step = 1.0 / max(self.num_stages, 1)

        self.skip_projectors = nn.ModuleList([
            FuseSkipProjector(skip_channels[idx], memory_channels) for idx in self.selected_indices
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(memory_channels, out_channels, kernel_size=1, bias=False),
            get_groupnorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_groupnorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def _stabilize(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=self.state_clip, neginf=-self.state_clip)
        return x.clamp(min=-self.state_clip, max=self.state_clip)

    def _ode_eq(self, x, y):
        return -y + torch.relu(x + y)

    def _step1_order1_explicit(self, x1, y1):
        f1 = self._ode_eq(x1, y1)
        y2 = y1 + self.step * f1
        return y2, f1

    def _step1_order2_implicit(self, x1, y1, x2):
        y2_pre, f1 = self._step1_order1_explicit(x1, y1)
        f2_pre = self._ode_eq(x2, y2_pre)
        y2 = y1 + (self.step / 2.0) * (f1 + f2_pre)
        return y2, f1

    def _steps2_order2_explicit(self, f1, x2, y2):
        f2 = self._ode_eq(x2, y2)
        y3 = y2 + (self.step / 2.0) * (3.0 * f2 - f1)
        return y3, f2

    def _steps2_order3_implicit(self, f1, x2, y2, x3):
        y3_pre, f2 = self._steps2_order2_explicit(f1, x2, y2)
        f3_pre = self._ode_eq(x3, y3_pre)
        y3 = y2 + (self.step / 12.0) * (5.0 * f3_pre + 8.0 * f2 - f1)
        return y3, f2

    def _steps3_order3_explicit(self, f1, f2, x3, y3):
        f3 = self._ode_eq(x3, y3)
        y4 = y3 + (self.step / 12.0) * (23.0 * f3 - 16.0 * f2 + 5.0 * f1)
        return y4, f3

    def _steps3_order4_implicit(self, f1, f2, x3, y3, x4):
        y4_pre, f3 = self._steps3_order3_explicit(f1, f2, x3, y3)
        f4_pre = self._ode_eq(x4, y4_pre)
        y4 = y3 + (self.step / 24.0) * (9.0 * f4_pre + 19.0 * f3 - 5.0 * f2 + f1)
        return y4, f3

    def _steps4_order4_explicit(self, f1, f2, f3, x4, y4):
        f4 = self._ode_eq(x4, y4)
        y5 = y4 + (self.step / 24.0) * (55.0 * f4 - 59.0 * f3 + 37.0 * f2 - 9.0 * f1)
        return y5, f4

    def _steps4_order4_implicit(self, f1, f2, f3, x4, y4, x5):
        y5_pre, f4 = self._steps4_order4_explicit(f1, f2, f3, x4, y4)
        f5_pre = self._ode_eq(x5, y5_pre)
        y5 = y4 + (self.step / 24.0) * (9.0 * f5_pre + 19.0 * f4 - 5.0 * f3 + f2)
        return y5, f4

    def _ab_predictor_update(self, current_stage, x_cur, y, f_hist):
        if current_stage == 1:
            return self._step1_order1_explicit(x_cur, y)
        if current_stage == 2:
            return self._steps2_order2_explicit(f_hist[0], x_cur, y)
        if current_stage == 3:
            return self._steps3_order3_explicit(f_hist[0], f_hist[1], x_cur, y)
        return self._steps4_order4_explicit(f_hist[-3], f_hist[-2], f_hist[-1], x_cur, y)

    def _am_corrector_update(self, current_stage, x_cur, y, f_hist):
        f_cur = self._ode_eq(x_cur, y)
        if current_stage == 1:
            y_next = y + self.step * f_cur
        elif current_stage == 2:
            y_next = y + (self.step / 12.0) * (5.0 * f_cur + 8.0 * f_hist[-1])
        else:
            y_next = y + (self.step / 24.0) * (9.0 * f_cur + 19.0 * f_hist[-1] - 5.0 * f_hist[-2])
        return y_next, f_cur

    def forward(self, skips):
        target_size = skips[self.target_index].shape[-2:]
        selected_skips = [skips[idx] for idx in self.selected_indices]
        projected = [proj(x, target_size) for proj, x in zip(self.skip_projectors, selected_skips)]

        # follow official indexing: x[-1] = deepest, x[0] = shallowest
        x = projected
        y = torch.zeros(
            x[0].shape[0],
            self.memory_channels,
            target_size[0],
            target_size[1],
            device=x[0].device,
            dtype=x[0].dtype,
        )
        y = self._stabilize(y)
        f_hist = []

        for current_stage in range(1, self.num_stages + 1):
            x_cur = x[-current_stage]
            if self.pcsc_mode == "ab_predictor":
                y, f_cur = self._ab_predictor_update(current_stage, x_cur, y, f_hist)
            elif self.pcsc_mode == "am_corrector":
                y, f_cur = self._am_corrector_update(current_stage, x_cur, y, f_hist)
            elif current_stage == self.num_stages:
                if current_stage == 1:
                    y, f_cur = self._step1_order1_explicit(x_cur, y)
                elif current_stage == 2:
                    y, f_cur = self._steps2_order2_explicit(f_hist[0], x_cur, y)
                elif current_stage == 3:
                    y, f_cur = self._steps3_order3_explicit(f_hist[0], f_hist[1], x_cur, y)
                else:
                    y, f_cur = self._steps4_order4_explicit(f_hist[0], f_hist[1], f_hist[2], x_cur, y)
            else:
                if current_stage == 1:
                    y, f_cur = self._step1_order2_implicit(x[-1], y, x[-2])
                elif current_stage == 2:
                    y, f_cur = self._steps2_order3_implicit(f_hist[0], x[-2], y, x[-3])
                elif current_stage == 3:
                    y, f_cur = self._steps3_order4_implicit(f_hist[0], f_hist[1], x[-3], y, x[-4])
                else:
                    y, f_cur = self._steps4_order4_implicit(
                        f_hist[-3], f_hist[-2], f_hist[-1], x[-current_stage], y, x[-current_stage - 1]
                    )
            y = self._stabilize(y)
            f_hist.append(self._stabilize(f_cur))
            if len(f_hist) > 4:
                f_hist = f_hist[-4:]

        out = self.out_proj(y)
        return self._stabilize(out)


class StaticSkipFusion(nn.Module):
    """Conventional projected multi-scale fusion used when PCSC is ablated."""

    def __init__(
        self,
        skip_channels,
        memory_channels,
        out_channels,
        selected_indices=(1, 2, 4),
        target_index=1,
    ):
        super().__init__()
        self.selected_indices = tuple(selected_indices)
        self.target_index = int(target_index)
        self.skip_projectors = nn.ModuleList([
            FuseSkipProjector(skip_channels[idx], memory_channels)
            for idx in self.selected_indices
        ])
        fused_channels = memory_channels * len(self.selected_indices)
        self.out_proj = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            get_groupnorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_groupnorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, skips):
        target_size = skips[self.target_index].shape[-2:]
        projected = [
            projector(skips[idx], target_size)
            for projector, idx in zip(self.skip_projectors, self.selected_indices)
        ]
        return self.out_proj(torch.cat(projected, dim=1))
