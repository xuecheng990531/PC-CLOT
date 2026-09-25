"""PC-PointCLOT main model — closed-loop transport refinement."""

import torch
import torch.nn as nn


class ClosedLoopTransportRefinementBlock(nn.Module):
    """Single iteration of the prediction-refinement loop.

    OT(point_maps, F, P, B) → T
    PrototypePooling(F, T) → R
    CrossAttentionRefinement(F, R) → F_next
    SharedHeads(F_next) → P_next, B_next
    """

    def __init__(self, ot_generator, prototype_pooling, refinement, heads):
        super().__init__()
        self.ot_generator = ot_generator
        self.prototype_pooling = prototype_pooling
        self.refinement = refinement
        self.heads = heads  # callable: (feature) -> (P, B)

    def forward(self, F, P, B, point_maps):
        # 1. OT: current prediction → transport assignment
        T = self.ot_generator(point_maps, F, P, B)

        # 2. Prototype pooling: transport → region prototypes
        R = self.prototype_pooling(F, T.detach(), B)

        # 3. Feature refinement: prototypes → corrected features
        F_next = self.refinement(F, R)

        # 4. Shared heads: refined features → refined predictions
        P_next, B_next = self.heads(F_next)

        return F_next, P_next, B_next, T, R


class PCPointCLOTLoop(nn.Module):
    """PC-PointCLOT with closed-loop transport refinement.

    Pipeline:
        backbone(image, point_maps) → F0, P0, B0
        for t in range(K):
            F, P, B, T, R = closed_loop_block(F, P, B, point_maps)
        → final P, B

    All K iterations share the same OTGenerator, PrototypePooling,
    CrossAttentionRefinement, and prediction heads.
    """

    def __init__(self, backbone, ot_generator, prototype_pooling,
                 refinement, num_iterations=2):
        super().__init__()
        self.backbone = backbone
        self.num_iterations = max(1, num_iterations)

        self.closed_loop_block = ClosedLoopTransportRefinementBlock(
            ot_generator=ot_generator,
            prototype_pooling=prototype_pooling,
            refinement=refinement,
            heads=backbone.predict_from_feature,
        )

    def advance_ot_step(self):
        self.closed_loop_block.ot_generator.advance_step()

    def forward(self, image, point_maps, point_coords=None, point_labels=None):
        # --- Initial backbone ---
        outputs0 = self.backbone(image, point_maps)

        F = outputs0["F0"]
        P = outputs0["mask_logits"]
        B = outputs0["boundary_logits"]

        history = {
            "F": [F],
            "P": [P],
            "B": [B],
            "T": [],
            "R": [],
        }

        # --- Iterative refinement ---
        for _ in range(self.num_iterations):
            F, P, B, T, R = self.closed_loop_block(F, P, B, point_maps)

            history["F"].append(F)
            history["P"].append(P)
            history["B"].append(B)
            history["T"].append(T)
            history["R"].append(R)

        return {
            "F0": history["F"][0],
            "P0": history["P"][0],
            "B0": history["B"][0],

            "F_final": history["F"][-1],
            "P_final": history["P"][-1],
            "B_final": history["B"][-1],
            "T_final": history["T"][-1] if history["T"] else None,

            "history": history,
            "num_iterations": self.num_iterations,

            "final_logits": history["P"][-1],
            "final_prob": torch.sigmoid(history["P"][-1]),

            # backward-compatible aliases
            "mask_logits0": history["P"][0],
            "boundary_logits0": history["B"][0],
            "mask_logits1": history["P"][1] if len(history["P"]) > 1 else history["P"][0],
            "boundary_logits1": history["B"][1] if len(history["B"]) > 1 else history["B"][0],
            "T1": history["T"][0] if history["T"] else None,
            "TR": history["T"][-1] if history["T"] else None,

            # For PrototypePooling backward compat (old code expects "prototypes")
            "prototypes": history["R"][-1] if history["R"] else None,
        }
