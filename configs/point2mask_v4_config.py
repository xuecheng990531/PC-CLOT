"""Config for V4 two-stage closed-loop OT refinement."""

from config import Config


class Point2MaskV4Config(Config):
    MODEL_NAME = "point2mask_v4"
    NUM_FG_POINTS = 1
    NUM_BG_POINTS = 1
    OT_ENABLE_V2 = True
    OT_USE_PREDICTION_COST = True
    MEMORY_CHANNELS = 64
    V4_SKIP_STAGE_INDICES = (1, 2, 4)
    V4_TARGET_STAGE_INDEX = 1
    USE_PCSC = True
    USE_PPOT = True
    V4_AUX_MASK_WEIGHT = 0.4
    USE_POINT_GUIDED_INTERACTION = False
    USE_CBAM = True
    POINT_GUIDE_CHANNELS = 32
    POINT_GUIDE_RESIDUAL_SCALE = 0.5
    CBAM_REDUCTION = 16

    V4_ENABLE = True
    V4_USE_BOUNDARY_PROTOTYPE = True
    V4_USE_UNCERTAINTY_PROTOTYPE = True
    V4_BOUNDARY_MIX = 0.5
    V4_UNCERTAINTY_SEM_WEIGHT = 0.5
    V4_UNCERTAINTY_MASK_WEIGHT = 0.5
    V4_STAGE2_WARMUP_STEPS = 1000
    V4_STAGE2_RAMP_STEPS = 2000
    V4_USE_PARTIAL_MASK_CE = True
    V4_USE_GENERALIZED_DICE = True
    V4_NOTES = (
        "Two-stage closed-loop OT refinement built on the representative-layer "
        "FuseUNet-style skip fusion backbone, with fg/bg plus boundary-aware and "
        "uncertainty-aware prototypes, optional SAM/MedSAM prior, plus CBAM."
    )
