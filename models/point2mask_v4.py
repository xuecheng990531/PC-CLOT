"""V4 Point2Mask branch with two-stage closed-loop OT refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    BoundaryHead,
    CBAM,
    DoubleConv,
    DownBlock,
    FuseUNetSkipFusion,
    MaskHead,
    PointGuidedInteraction,
    SemanticHead,
    StaticSkipFusion,
)
from .prototype_pooling import PrototypePooling
from .clot_refinement import CrossAttentionRefinement
from .point2mask_ot import Point2MaskOT, _stabilize_tensor


class FuseUNetEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=None):
        super().__init__()
        channels = channels or [64, 128, 256, 512, 512]
        self.stem = DoubleConv(in_channels, channels[0])
        self.enc1 = DownBlock(channels[0], channels[1])
        self.enc2 = DownBlock(channels[1], channels[2])
        self.enc3 = DownBlock(channels[2], channels[3])
        self.enc4 = DownBlock(channels[3], channels[4])

    def forward(self, image):
        image = _stabilize_tensor(image, clamp_value=5.0)
        x0 = _stabilize_tensor(self.stem(image))
        x1 = _stabilize_tensor(self.enc1(x0))
        x2 = _stabilize_tensor(self.enc2(x1))
        x3 = _stabilize_tensor(self.enc3(x2))
        x4 = _stabilize_tensor(self.enc4(x3))
        return [x0, x1, x2, x3, x4]


class Point2MaskPolypV4(nn.Module):
    def __init__(
        self,
        backbone_channels=None,
        memory_channels=32,
        pc_delta=0.1,
        pc_residual_scale=0.1,
        pc_state_clip=10.0,
        use_pcskipfusion=False,
        ot_downsample=4,
        ot_spatial_distance_mode="dijkstra",
        ot_geodesic_neighborhood=8,
        ot_path_cost_mode="learned",
        ot_use_boundary_barrier=True,
        ot_lambda_prob=1.0,
        ot_lambda_boundary=0.4,
        ot_lambda_feat=0.2,
        ot_lambda_edge=0.0,
        ot_sinkhorn_eps=0.05,
        ot_sinkhorn_iters=60,
        use_low_level_edge=False,
        ot_warmup_steps=1500,
        ot_warmup_fg_ratio=0.08,
        ot_min_fg_ratio=0.03,
        ot_max_fg_ratio=0.30,
        ot_semantic_ramp_steps=2500,
        ot_enable_v2=True,
        ot_fg_boundary_gamma=2.0,
        ot_use_prediction_cost=True,
        ot_pred_cost_weight=0.35,
        ot_pred_warmup_steps=1000,
        ot_pred_ramp_steps=2500,
        ot_unary_semantic_weight=0.75,
        ot_unary_boundary_weight=0.40,
        ot_pseudo_mask_blend_weight=0.50,
        ot_use_bg_exclusion=False,
        ot_bg_exclusion_weight=0.75,
        ot_bg_exclusion_tau=0.20,
        ot_use_sam_prior=False,
        ot_sam_prior_weight=0.20,
        ot_sam_prior_warmup_steps=1500,
        ot_sam_prior_ramp_steps=3000,
        ot_sam_prior_mass_weight=0.25,
        skip_stage_indices=(1, 2, 4),
        target_stage_index=1,
        use_pcsc=True,
        pcsc_design="full",
        use_ppot=True,
        v4_use_boundary_prototype=True,
        v4_use_uncertainty_prototype=True,
        v4_boundary_mix=0.5,
        v4_uncertainty_sem_weight=0.5,
        v4_uncertainty_mask_weight=0.5,
        v4_stage2_warmup_steps=1000,
        v4_stage2_ramp_steps=2000,
        refinement_temperature=1.0,
        refinement_use_projection=True,
        refinement_residual_scale=1.0,
        closed_loop_ablation="closed_loop_tipr",
        use_point_guided_interaction=True,
        use_cbam=True,
        point_guide_channels=32,
        point_guide_residual_scale=0.5,
        cbam_reduction=16,
    ):
        super().__init__()
        del pc_residual_scale, use_pcskipfusion
        backbone_channels = backbone_channels or [64, 128, 256, 512, 512]
        head_channels = backbone_channels[0]

        self.v_stage = "v4_two_stage_closed_loop"
        self.closed_loop_ablation = closed_loop_ablation
        self.use_ppot = use_ppot
        self.use_stage2_transport = use_ppot and closed_loop_ablation in {"closed_loop_no_proto", "closed_loop_tipr"}
        self.use_prototype_feedback = use_ppot and closed_loop_ablation == "closed_loop_tipr"
        # The backbone consumes one five-channel tensor:
        # RGB (3) + foreground clicks (1) + background clicks (1).
        self.input_channels = 5
        self.encoder = FuseUNetEncoder(in_channels=self.input_channels, channels=backbone_channels)
        self.use_pcsc = use_pcsc
        self.pcsc_design = pcsc_design if use_pcsc else "vanilla"
        if use_pcsc:
            self.skip_fusion = FuseUNetSkipFusion(
                skip_channels=backbone_channels,
                memory_channels=memory_channels,
                out_channels=head_channels,
                delta=pc_delta,
                state_clip=pc_state_clip,
                selected_indices=skip_stage_indices,
                target_index=target_stage_index,
                pcsc_mode=pcsc_design,
            )
        else:
            self.skip_fusion = StaticSkipFusion(
                skip_channels=backbone_channels,
                memory_channels=memory_channels,
                out_channels=head_channels,
                selected_indices=skip_stage_indices,
                target_index=target_stage_index,
            )
        self.feature_refine = DoubleConv(head_channels, head_channels)
        self.use_point_guided_interaction = use_point_guided_interaction
        self.point_guided_interaction = PointGuidedInteraction(
            feature_channels=head_channels,
            guide_channels=point_guide_channels,
            residual_scale=point_guide_residual_scale,
        ) if use_point_guided_interaction else nn.Identity()
        self.cbam = CBAM(head_channels, reduction=cbam_reduction) if use_cbam else nn.Identity()
        self.prototype_names = ["bg", "fg"]
        if v4_use_boundary_prototype:
            self.prototype_names.append("boundary")
        if v4_use_uncertainty_prototype:
            self.prototype_names.append("uncertainty")
        self.prototype_pooling = PrototypePooling(
            use_boundary_prototype=v4_use_boundary_prototype,
            use_uncertainty_prototype=v4_use_uncertainty_prototype,
            boundary_mix=v4_boundary_mix,
            uncertainty_sem_weight=v4_uncertainty_sem_weight,
            uncertainty_mask_weight=v4_uncertainty_mask_weight,
            return_dict=True,
        )
        self.refinement = CrossAttentionRefinement(
            channels=head_channels,
            num_prototypes=len(self.prototype_names),
            temperature=refinement_temperature,
            use_projection=refinement_use_projection,
            residual_scale=refinement_residual_scale,
        )
        self.v4_stage2_warmup_steps = v4_stage2_warmup_steps
        self.v4_stage2_ramp_steps = v4_stage2_ramp_steps
        self.semantic_head = SemanticHead(in_channels=head_channels, num_classes=2)
        self.boundary_head = BoundaryHead(in_channels=head_channels)
        self.mask_head = MaskHead(in_channels=head_channels)
        self.ot = Point2MaskOT(
            downsample=ot_downsample,
            spatial_distance_mode=ot_spatial_distance_mode,
            geodesic_neighborhood=ot_geodesic_neighborhood,
            path_cost_mode=ot_path_cost_mode,
            use_boundary_barrier=ot_use_boundary_barrier,
            lambda_prob=ot_lambda_prob,
            lambda_boundary=ot_lambda_boundary,
            lambda_feat=ot_lambda_feat,
            lambda_edge=ot_lambda_edge,
            sinkhorn_eps=ot_sinkhorn_eps,
            sinkhorn_iters=ot_sinkhorn_iters,
            use_low_level_edge=use_low_level_edge,
            warmup_steps=ot_warmup_steps,
            warmup_fg_ratio=ot_warmup_fg_ratio,
            min_fg_ratio=ot_min_fg_ratio,
            max_fg_ratio=ot_max_fg_ratio,
            semantic_ramp_steps=ot_semantic_ramp_steps,
            enable_v2=ot_enable_v2,
            fg_boundary_gamma=ot_fg_boundary_gamma,
            use_prediction_cost=ot_use_prediction_cost,
            pred_cost_weight=ot_pred_cost_weight,
            pred_warmup_steps=ot_pred_warmup_steps,
            pred_ramp_steps=ot_pred_ramp_steps,
            unary_semantic_weight=ot_unary_semantic_weight,
            unary_boundary_weight=ot_unary_boundary_weight,
            pseudo_mask_blend_weight=ot_pseudo_mask_blend_weight,
            use_bg_exclusion=ot_use_bg_exclusion,
            bg_exclusion_weight=ot_bg_exclusion_weight,
            bg_exclusion_tau=ot_bg_exclusion_tau,
            use_sam_prior=ot_use_sam_prior,
            sam_prior_weight=ot_sam_prior_weight,
            sam_prior_warmup_steps=ot_sam_prior_warmup_steps,
            sam_prior_ramp_steps=ot_sam_prior_ramp_steps,
            sam_prior_mass_weight=ot_sam_prior_mass_weight,
        )

    def advance_ot_step(self):
        self.ot.advance_step()

    def _stage2_weight(self):
        step = int(self.ot._step.item())
        if step < self.v4_stage2_warmup_steps:
            return 0.0
        if self.v4_stage2_ramp_steps <= 0:
            return 1.0
        ramp_step = min(step - self.v4_stage2_warmup_steps, self.v4_stage2_ramp_steps)
        return float(ramp_step) / float(self.v4_stage2_ramp_steps)

    def _predict_from_feature(self, feature):
        semantic_logits = _stabilize_tensor(self.semantic_head(feature))
        boundary_logits = _stabilize_tensor(self.boundary_head(feature))
        mask_logits = _stabilize_tensor(self.mask_head(feature))
        return semantic_logits, boundary_logits, mask_logits

    def _encode_feature(self, image, point_maps):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected RGB image [B, 3, H, W], got {tuple(image.shape)}")
        if point_maps.ndim != 4 or point_maps.shape[1] != 2:
            raise ValueError(
                "Expected click maps [B, 2, H, W] ordered as foreground/background, "
                f"got {tuple(point_maps.shape)}"
            )
        if image.shape[0] != point_maps.shape[0] or image.shape[-2:] != point_maps.shape[-2:]:
            raise ValueError(
                "RGB image and click maps must have matching batch/spatial dimensions: "
                f"image={tuple(image.shape)}, point_maps={tuple(point_maps.shape)}"
            )

        model_input = torch.cat([image, point_maps.to(dtype=image.dtype)], dim=1)
        skips = self.encoder(model_input)
        feature = _stabilize_tensor(self.skip_fusion(skips))
        if feature.shape[-2:] != image.shape[-2:]:
            feature = F.interpolate(feature, size=image.shape[-2:], mode="bilinear", align_corners=False)
        feature = _stabilize_tensor(self.feature_refine(feature))
        if self.use_point_guided_interaction:
            feature = _stabilize_tensor(self.point_guided_interaction(feature, image, point_maps))
        feature = _stabilize_tensor(self.cbam(feature))
        return feature

    def forward(self, image, point_maps, point_coords=None, point_labels=None, sam_prior=None):
        if point_coords is None or point_labels is None:
            raise ValueError("V4 OT requires point_coords and point_labels from the dataset.")

        F0 = self._encode_feature(image, point_maps)
        Ps0, Pb0, P0 = self._predict_from_feature(F0)
        if self.use_ppot:
            T1, pseudo_mask0, ot_stats0 = self.ot(
                image=image, feature=F0, semantic_logits=Ps0, boundary_logits=Pb0, mask_logits=P0,
                point_coords=point_coords, point_labels=point_labels, point_maps=point_maps,
                sam_prior=sam_prior,
            )
        else:
            pseudo_mask0 = torch.sigmoid(P0)
            T1 = torch.cat([1.0 - pseudo_mask0, pseudo_mask0], dim=1)
            ot_stats0 = {}

        if self.use_prototype_feedback:
            proto_outputs = self.prototype_pooling(
                F0,
                T1.detach(),
                boundary_logits=Pb0.detach(),
                semantic_logits=Ps0.detach(),
                mask_logits=P0.detach(),
            )
            prototypes = proto_outputs["prototypes"]
            F1 = _stabilize_tensor(self.refinement(F0, prototypes))
            Ps1, Pb1, P1 = self._predict_from_feature(F1)
        else:
            proto_outputs = {
                "prototypes": None,
                "prototype_names": [],
                "weight_maps": {},
                "named_prototypes": {},
            }
            prototypes = None
            F1 = F0
            Ps1, Pb1, P1 = Ps0, Pb0, P0

        if self.use_stage2_transport:
            TR, pseudo_mask1, ot_stats1 = self.ot(
                image=image, feature=F1, semantic_logits=Ps1, boundary_logits=Pb1, mask_logits=P1,
                point_coords=point_coords, point_labels=point_labels, point_maps=point_maps,
                sam_prior=sam_prior,
            )
        else:
            TR, pseudo_mask1, ot_stats1 = T1, pseudo_mask0, ot_stats0
        stage2_weight = self._stage2_weight() if self.use_stage2_transport else 0.0
        if self.use_stage2_transport:
            stable_logits = (1.0 - stage2_weight) * P0 + stage2_weight * P1
            final_logits = P1
        else:
            stable_logits = P0
            final_logits = P0

        return {
            "image": image,
            "F0": F0, "F1": F1, "feature": F1,
            "semantic_logits0": Ps0, "semantic_logits": Ps1,
            "semantic_prob0": torch.softmax(Ps0, dim=1), "semantic_prob": torch.softmax(Ps1, dim=1),
            "boundary_logits0": Pb0, "boundary_logits": Pb1,
            "boundary_prob0": torch.sigmoid(Pb0), "boundary_prob": torch.sigmoid(Pb1),
            "mask_logits0": P0, "mask_logits": P1,
            "mask_prob0": torch.sigmoid(P0), "mask_prob": torch.sigmoid(P1),
            "transport0": T1, "transport": TR,
            "pseudo_mask0": pseudo_mask0, "pseudo_mask": pseudo_mask1,
            "ot_stats0": ot_stats0, "ot_stats": ot_stats1,
            "prototypes": prototypes,
            "prototype_names": proto_outputs["prototype_names"],
            "prototype_maps": proto_outputs["weight_maps"],
            "named_prototypes": proto_outputs["named_prototypes"],
            "history": {"F": [F0, F1], "P": [P0, P1], "B": [Pb0, Pb1], "T": [T1, TR], "Ps": [Ps0, Ps1], "pseudo_mask": [pseudo_mask0, pseudo_mask1]},
            "stable_logits": stable_logits,
            "stable_prob": torch.sigmoid(stable_logits),
            "final_logits": final_logits,
            "final_prob": torch.sigmoid(final_logits),
            "pred_fg_ratio": torch.sigmoid(final_logits).mean(),
            "v4_stage2_weight": final_logits.new_tensor(stage2_weight),
            "T_final": TR, "T1": T1, "TR": TR,
            "closed_loop_ablation": self.closed_loop_ablation,
            "use_pcsc": self.use_pcsc,
            "pcsc_design": self.pcsc_design,
            "use_ppot": self.use_ppot,
        }
