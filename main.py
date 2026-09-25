"""Point2Mask V4 main entry point."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Point2MaskPolypV4, Point2MaskPolypV4SAM, Point2MaskPolypV4SynFoC
from data import create_dataloaders, discover_samples
from train import Trainer
from utils import create_optimizer, create_scheduler, print_model_summary
from config import Config
from configs import Point2MaskV4Config
from ablations.ppot_cost_components import (
    PPOT_COST_COMPONENT_VARIANTS,
    get_ppot_cost_component_preset,
)


VALID_DATASETS = ['cvc', 'kvasir', 'cvc_clon']
VALID_MODELS = ['point2mask_v4', 'point2mask_v4_sam', 'point2mask_v4_synfoc']
CLOSED_LOOP_ABLATIONS = ['one_pass_ppot', 'closed_loop_no_proto', 'closed_loop_tipr']
PROTOTYPE_ABLATIONS = ['fg_bg', 'fg_bg_boundary', 'fg_bg_uncertainty', 'fg_bg_boundary_uncertainty']
PPOT_COST_ABLATIONS = list(PPOT_COST_COMPONENT_VARIANTS.keys())
PCSC_DESIGNS = ['full', 'ab_predictor', 'am_corrector']


def _sinkhorn_iter_tag(ot_iters):
    default_iters = getattr(Config, 'OT_DEFAULT_NUM_ITERS', Config.OT_NUM_ITERS)
    if ot_iters == default_iters:
        return None
    return f"sinkhorn_iter_{ot_iters}"


def _geodesic_cost_tag(args):
    if args.ot_spatial_distance_mode == 'euclidean':
        return 'euclidean_distance'
    default_neighborhood = getattr(Config, 'OT_GEODESIC_NEIGHBORHOOD', 8)
    default_path_mode = getattr(Config, 'OT_PATH_COST_MODE', 'learned')
    default_boundary_barrier = getattr(Config, 'OT_USE_BOUNDARY_BARRIER', True)
    neighborhood = args.ot_geodesic_neighborhood
    path_mode = args.ot_path_cost_mode
    use_boundary_barrier = not args.ot_disable_boundary_barrier
    if (
        neighborhood == default_neighborhood
        and path_mode == default_path_mode
        and use_boundary_barrier == default_boundary_barrier
    ):
        return None
    if path_mode == 'uniform':
        return f'{neighborhood}n_dijkstra'
    if not use_boundary_barrier:
        return 'dijkstra_no_boundary_barrier'
    return f'geodesic_{neighborhood}n_{path_mode}'


def parse_args():
    p = argparse.ArgumentParser(description='PC-PointCLOT: Point-guided Polyp Segmentation')

    # --- Mode ---
    p.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                   help='train or test')

    # --- Model ---
    p.add_argument('--model', type=str, default=Config.MODEL_NAME, choices=VALID_MODELS,
                   help='Training path to use.')
    p.add_argument('--closed_loop_ablation', type=str, default=Config.CLOSED_LOOP_ABLATION,
                   choices=CLOSED_LOOP_ABLATIONS,
                   help='Closed-loop ablation variant for Table: one_pass_ppot, closed_loop_no_proto, closed_loop_tipr.')
    p.add_argument('--prototype_ablation', type=str, default=Config.PROTOTYPE_ABLATION,
                   choices=PROTOTYPE_ABLATIONS,
                   help='Prototype composition ablation: fg_bg, fg_bg_boundary, fg_bg_uncertainty, fg_bg_boundary_uncertainty.')
    p.add_argument('--disable_pcsc', action='store_true',
                   help='Replace predictor-corrector skip calibration with static projected multi-scale fusion.')
    p.add_argument('--pcsc_design', type=str, default=Config.PCSC_DESIGN,
                   choices=PCSC_DESIGNS,
                   help='PCSC design variant: full, ab_predictor, or am_corrector. Ignored when --disable_pcsc is set.')
    p.add_argument('--disable_ppot', action='store_true',
                   help='Disable OT pseudo-mask generation for a point-only baseline.')
    p.add_argument('--ppot_cost_ablation', type=str, default=Config.PPOT_COST_ABLATION,
                   choices=PPOT_COST_ABLATIONS,
                   help='PPOT cost ablation: spatial_distance_only, spatial_semantic, spatial_semantic_boundary, spatial_semantic_boundary_feature.')

    # --- Dataset ---
    p.add_argument('--dataset', type=str, required=True, choices=VALID_DATASETS,
                   help='Dataset to use (cvc, kvasir, or cvc_clon). Datasets are never mixed.')
    p.add_argument('--data_root', type=str, default='/icislab/volume1/lxc/polyp_data',
                   help='Root directory containing cvc/ and kvasir/ subdirectories')
    p.add_argument('--split_dir', type=str, default='/icislab/volume1/lxc/polyp_data/splits',
                   help='Directory to save/load split JSON files')

    # --- Split control ---
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--test_ratio', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--regenerate_splits', action='store_true',
                   help='Force re-generation of split file for the selected dataset')

    # --- Training ---
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=Config.LR_MIN)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--grad_accum_steps', type=int, default=Config.GRAD_ACCUM_STEPS)

    # --- Data ---
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--num_fg_points', type=int, default=1,
                   help='Number of foreground clicks per image (default: 1).')
    p.add_argument('--num_bg_points', type=int, default=1,
                   help='Number of background clicks per image (default: 1).')
    p.add_argument('--min_point_distance', type=int, default=20)
    p.add_argument('--point_radius', type=int, default=10)
    p.add_argument('--test_noise_std', type=float, default=0.0,
                   help='Gaussian noise std added only to test images after normalization.')

    # --- Legacy PC-PointCLOT model ---
    p.add_argument('--memory_channels', type=int, default=Config.MEMORY_CHANNELS)
    p.add_argument('--num_iterations', type=int, default=Config.NUM_ITERATIONS,
                   help='Number of closed-loop refinement iterations (K)')

    # --- OT ---
    p.add_argument('--ot_iters', type=int, default=Config.OT_NUM_ITERS,
                   help='Number of Sinkhorn iterations per OT call')
    p.add_argument('--ot_temperature', type=float, default=Config.OT_TEMPERATURE,
                   help='Sinkhorn temperature')
    p.add_argument('--ot_downsample', type=int, default=Config.OT_DOWNSAMPLE,
                   help='Downsample factor used by Point2Mask-style shortest-path OT')
    p.add_argument('--ot_spatial_distance_mode', type=str, default=Config.OT_SPATIAL_DISTANCE_MODE,
                   choices=['euclidean', 'dijkstra'],
                   help='Spatial distance backend used inside OT.')
    p.add_argument('--ot_geodesic_neighborhood', type=int, default=Config.OT_GEODESIC_NEIGHBORHOOD,
                   choices=[4, 8],
                   help='Neighborhood used by graph shortest-path OT.')
    p.add_argument('--ot_path_cost_mode', type=str, default=Config.OT_PATH_COST_MODE,
                   choices=['uniform', 'learned'],
                   help='Whether Dijkstra uses uniform grid costs or learned geodesic edge costs.')
    p.add_argument('--ot_disable_boundary_barrier', action='store_true',
                   help='Disable boundary-aware barrier in the geodesic path cost while keeping other OT terms unchanged.')
    p.add_argument('--use_low_level_edge', action='store_true',
                   help='Enable a simple Sobel low-level edge prior in the OT cost')
    p.add_argument('--use_sam_prior', action='store_true',
                   help='Enable SAM/MedSAM soft-mask prior in the OT unary cost')
    p.add_argument('--sam_prior_dir', type=str, default=Config.SAM_PRIOR_DIR,
                   help='Directory containing offline SAM/MedSAM priors named by sample stem')
    p.add_argument('--sam_prior_required', action='store_true',
                   help='Raise an error if --sam_prior_dir is enabled but a sample prior is missing')
    p.add_argument('--sam_prior_weight', type=float, default=Config.OT_SAM_PRIOR_WEIGHT,
                   help='Final unary-cost weight for SAM/MedSAM prior')
    p.add_argument('--sam_prior_warmup_steps', type=int, default=Config.OT_SAM_PRIOR_WARMUP_STEPS,
                   help='Optimizer steps before SAM/MedSAM prior starts affecting OT')
    p.add_argument('--sam_prior_ramp_steps', type=int, default=Config.OT_SAM_PRIOR_RAMP_STEPS,
                   help='Steps used to ramp SAM/MedSAM prior cost weight')
    p.add_argument('--sam_prior_mass_weight', type=float, default=Config.OT_SAM_PRIOR_MASS_WEIGHT,
                   help='Blend weight for using prior mean as OT foreground mass target')
    p.add_argument('--cps_sam_teacher_steps', type=int, default=Config.CPS_SAM_TEACHER_STEPS,
                   help='Steps over which CPS accepts SAM-only regions as early teacher evidence')
    p.add_argument('--cps_net_correction_warmup_steps', type=int, default=Config.CPS_NET_CORRECTION_WARMUP_STEPS,
                   help='Warmup steps before CPS accepts network-only regions to correct SAM')
    p.add_argument('--cps_net_correction_ramp_steps', type=int, default=Config.CPS_NET_CORRECTION_RAMP_STEPS,
                   help='Ramp steps for CPS network correction evidence')
    p.add_argument('--cps_point_kernel', type=int, default=Config.CPS_POINT_KERNEL,
                   help='Max-pooling kernel used to expand foreground point support in CPS')
    p.add_argument('--cps_boundary_weight', type=float, default=Config.CPS_BOUNDARY_WEIGHT,
                   help='Boundary penalty weight used when accepting divergent CPS regions')
    p.add_argument('--synfoc_weight', type=float, default=Config.SYNFOC_WEIGHT,
                   help='OT cost weight for SynFoC-style consensus/divergence correction')
    p.add_argument('--synfoc_sam_teacher_steps', type=int, default=Config.SYNFOC_SAM_TEACHER_STEPS,
                   help='Steps over which SAM-only evidence decays as early teacher')
    p.add_argument('--synfoc_net_correction_warmup_steps', type=int, default=Config.SYNFOC_NET_CORRECTION_WARMUP_STEPS,
                   help='Warmup steps before network-only evidence can correct SAM')
    p.add_argument('--synfoc_net_correction_ramp_steps', type=int, default=Config.SYNFOC_NET_CORRECTION_RAMP_STEPS,
                   help='Ramp steps for network correction evidence')
    p.add_argument('--synfoc_mass_weight', type=float, default=Config.SYNFOC_MASS_WEIGHT,
                   help='Blend weight for SynFoC foreground evidence mean in OT mass target')

    # --- Save ---
    p.add_argument('--save_dir', type=str, default='runs/pc_pointclot')
    p.add_argument('--save_val_visuals', action='store_true', default=True)
    p.add_argument('--num_val_visuals', type=int, default=5)
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to checkpoint for testing')
    p.add_argument('--test_sample_name', type=str, default=None,
                   help='If set in test mode, evaluate only the matching sample stem or filename.')
    p.add_argument('--test_full_dataset', action='store_true',
                   help='In test mode, evaluate every image in the selected dataset instead of its saved test split.')
    p.add_argument('--eval_seeds', type=int, nargs='+', default=None,
                   help='Seeds used for final repeated evaluation. Default: seed, seed+1, ..., seed+4.')

    # --- SwanLab ---
    p.add_argument('--use_swanlab', action='store_true')
    p.add_argument('--swanlab_project', type=str, default='PC-PointCLOT-Polyp')
    p.add_argument('--swanlab_experiment', type=str, default=None)

    return p.parse_args()


def build_model(config, device):
    prototype_ablation = getattr(config, 'PROTOTYPE_ABLATION', 'fg_bg_boundary_uncertainty')
    use_boundary_prototype = prototype_ablation in {'fg_bg_boundary', 'fg_bg_boundary_uncertainty'}
    use_uncertainty_prototype = prototype_ablation in {'fg_bg_uncertainty', 'fg_bg_boundary_uncertainty'}
    model_name = getattr(config, 'MODEL_NAME', 'point2mask_v4')
    model_cls = {
        'point2mask_v4': Point2MaskPolypV4,
        'point2mask_v4_sam': Point2MaskPolypV4SAM,
        'point2mask_v4_synfoc': Point2MaskPolypV4SynFoC,
    }[model_name]
    model_kwargs = dict(
        backbone_channels=config.ENCODER_CHANNELS,
        memory_channels=config.MEMORY_CHANNELS,
        pc_delta=config.PC_DELTA,
        pc_residual_scale=config.PC_RESIDUAL_SCALE,
        pc_state_clip=config.PC_STATE_CLIP,
        ot_downsample=config.OT_DOWNSAMPLE,
        ot_spatial_distance_mode=getattr(config, 'OT_SPATIAL_DISTANCE_MODE', 'dijkstra'),
        ot_geodesic_neighborhood=getattr(config, 'OT_GEODESIC_NEIGHBORHOOD', 8),
        ot_path_cost_mode=getattr(config, 'OT_PATH_COST_MODE', 'learned'),
        ot_use_boundary_barrier=getattr(config, 'OT_USE_BOUNDARY_BARRIER', True),
        ot_lambda_prob=config.OT_LAMBDA_PROB,
        ot_lambda_boundary=config.OT_LAMBDA_BOUNDARY,
        ot_lambda_feat=config.OT_LAMBDA_FEAT,
        ot_lambda_edge=config.OT_LAMBDA_EDGE,
        ot_sinkhorn_eps=config.OT_SINKHORN_EPS,
        ot_sinkhorn_iters=config.OT_NUM_ITERS,
        use_low_level_edge=getattr(config, 'OT_USE_LOW_LEVEL_EDGE', False),
        ot_warmup_steps=config.OT_WARMUP_STEPS,
        ot_warmup_fg_ratio=config.OT_WARMUP_FG_RATIO,
        ot_min_fg_ratio=config.OT_MIN_FG_RATIO,
        ot_max_fg_ratio=config.OT_MAX_FG_RATIO,
        ot_semantic_ramp_steps=config.OT_SEMANTIC_RAMP_STEPS,
        ot_enable_v2=True,
        ot_fg_boundary_gamma=config.OT_FG_BOUNDARY_GAMMA,
        ot_use_prediction_cost=config.OT_USE_PREDICTION_COST,
        ot_pred_cost_weight=config.OT_PRED_COST_WEIGHT,
        ot_pred_warmup_steps=config.OT_PRED_WARMUP_STEPS,
        ot_pred_ramp_steps=config.OT_PRED_RAMP_STEPS,
        ot_unary_semantic_weight=config.OT_UNARY_SEMANTIC_WEIGHT,
        ot_unary_boundary_weight=config.OT_UNARY_BOUNDARY_WEIGHT,
        ot_pseudo_mask_blend_weight=config.OT_PSEUDO_MASK_BLEND_WEIGHT,
        ot_use_bg_exclusion=config.OT_USE_BG_EXCLUSION,
        ot_bg_exclusion_weight=config.OT_BG_EXCLUSION_WEIGHT,
        ot_bg_exclusion_tau=config.OT_BG_EXCLUSION_TAU,
        ot_use_sam_prior=getattr(config, 'OT_USE_SAM_PRIOR', False),
        ot_sam_prior_weight=getattr(config, 'OT_SAM_PRIOR_WEIGHT', 0.20),
        ot_sam_prior_warmup_steps=getattr(config, 'OT_SAM_PRIOR_WARMUP_STEPS', 1500),
        ot_sam_prior_ramp_steps=getattr(config, 'OT_SAM_PRIOR_RAMP_STEPS', 3000),
        ot_sam_prior_mass_weight=getattr(config, 'OT_SAM_PRIOR_MASS_WEIGHT', 0.25),
        skip_stage_indices=getattr(config, 'V4_SKIP_STAGE_INDICES', (1, 2, 4)),
        target_stage_index=getattr(config, 'V4_TARGET_STAGE_INDEX', 1),
        use_pcsc=getattr(config, 'USE_PCSC', True),
        pcsc_design=getattr(config, 'PCSC_DESIGN', 'full'),
        use_ppot=getattr(config, 'USE_PPOT', True),
        v4_use_boundary_prototype=use_boundary_prototype,
        v4_use_uncertainty_prototype=use_uncertainty_prototype,
        v4_boundary_mix=getattr(config, 'V4_BOUNDARY_MIX', 0.5),
        v4_uncertainty_sem_weight=getattr(config, 'V4_UNCERTAINTY_SEM_WEIGHT', 0.5),
        v4_uncertainty_mask_weight=getattr(config, 'V4_UNCERTAINTY_MASK_WEIGHT', 0.5),
        v4_stage2_warmup_steps=getattr(config, 'V4_STAGE2_WARMUP_STEPS', 1000),
        v4_stage2_ramp_steps=getattr(config, 'V4_STAGE2_RAMP_STEPS', 2000),
        refinement_temperature=config.REFINEMENT_TEMPERATURE,
        refinement_use_projection=config.REFINEMENT_USE_PROJECTION,
        refinement_residual_scale=config.REFINEMENT_RESIDUAL_SCALE,
        closed_loop_ablation=getattr(config, 'CLOSED_LOOP_ABLATION', 'closed_loop_tipr'),
        use_point_guided_interaction=getattr(config, 'USE_POINT_GUIDED_INTERACTION', False),
        use_cbam=getattr(config, 'USE_CBAM', True),
        point_guide_channels=getattr(config, 'POINT_GUIDE_CHANNELS', 32),
        point_guide_residual_scale=getattr(config, 'POINT_GUIDE_RESIDUAL_SCALE', 0.5),
        cbam_reduction=getattr(config, 'CBAM_REDUCTION', 16),
    )
    if model_name == 'point2mask_v4_synfoc':
        model_kwargs.update(
            synfoc_weight=getattr(config, 'SYNFOC_WEIGHT', 0.20),
            synfoc_sam_teacher_steps=getattr(config, 'SYNFOC_SAM_TEACHER_STEPS', 3000),
            synfoc_net_correction_warmup_steps=getattr(config, 'SYNFOC_NET_CORRECTION_WARMUP_STEPS', 1500),
            synfoc_net_correction_ramp_steps=getattr(config, 'SYNFOC_NET_CORRECTION_RAMP_STEPS', 3000),
            synfoc_mass_weight=getattr(config, 'SYNFOC_MASS_WEIGHT', 0.25),
        )
    if model_name in {'point2mask_v4_sam', 'point2mask_v4_synfoc'}:
        model_kwargs.update(
            cps_sam_teacher_steps=getattr(config, 'CPS_SAM_TEACHER_STEPS', 3000),
            cps_net_correction_warmup_steps=getattr(config, 'CPS_NET_CORRECTION_WARMUP_STEPS', 1500),
            cps_net_correction_ramp_steps=getattr(config, 'CPS_NET_CORRECTION_RAMP_STEPS', 3000),
            cps_point_kernel=getattr(config, 'CPS_POINT_KERNEL', 31),
            cps_boundary_weight=getattr(config, 'CPS_BOUNDARY_WEIGHT', 1.0),
        )
    model = model_cls(**model_kwargs).to(device)
    return model


def apply_overrides(config, args):
    """Apply CLI argument overrides to Config defaults."""
    config.MODEL_NAME = args.model
    config.MEMORY_CHANNELS = args.memory_channels
    config.NUM_ITERATIONS = args.num_iterations
    config.OT_NUM_ITERS = args.ot_iters
    config.OT_TEMPERATURE = args.ot_temperature
    config.OT_DOWNSAMPLE = args.ot_downsample
    config.OT_SPATIAL_DISTANCE_MODE = args.ot_spatial_distance_mode
    config.OT_GEODESIC_NEIGHBORHOOD = args.ot_geodesic_neighborhood
    config.OT_PATH_COST_MODE = args.ot_path_cost_mode
    config.OT_USE_BOUNDARY_BARRIER = not args.ot_disable_boundary_barrier
    config.OT_USE_LOW_LEVEL_EDGE = args.use_low_level_edge
    config.OT_USE_SAM_PRIOR = bool(args.use_sam_prior or args.sam_prior_dir)
    config.SAM_PRIOR_DIR = args.sam_prior_dir
    config.SAM_PRIOR_REQUIRED = args.sam_prior_required
    config.OT_SAM_PRIOR_WEIGHT = args.sam_prior_weight
    config.OT_SAM_PRIOR_WARMUP_STEPS = args.sam_prior_warmup_steps
    config.OT_SAM_PRIOR_RAMP_STEPS = args.sam_prior_ramp_steps
    config.OT_SAM_PRIOR_MASS_WEIGHT = args.sam_prior_mass_weight
    config.CPS_SAM_TEACHER_STEPS = args.cps_sam_teacher_steps
    config.CPS_NET_CORRECTION_WARMUP_STEPS = args.cps_net_correction_warmup_steps
    config.CPS_NET_CORRECTION_RAMP_STEPS = args.cps_net_correction_ramp_steps
    config.CPS_POINT_KERNEL = args.cps_point_kernel
    config.CPS_BOUNDARY_WEIGHT = args.cps_boundary_weight
    config.SYNFOC_WEIGHT = args.synfoc_weight
    config.SYNFOC_SAM_TEACHER_STEPS = args.synfoc_sam_teacher_steps
    config.SYNFOC_NET_CORRECTION_WARMUP_STEPS = args.synfoc_net_correction_warmup_steps
    config.SYNFOC_NET_CORRECTION_RAMP_STEPS = args.synfoc_net_correction_ramp_steps
    config.SYNFOC_MASS_WEIGHT = args.synfoc_mass_weight
    config.NUM_FG_POINTS = args.num_fg_points
    config.NUM_BG_POINTS = args.num_bg_points
    config.SEED = args.seed
    config.GRAD_ACCUM_STEPS = args.grad_accum_steps
    config.LR_MIN = args.min_lr
    config.CLOSED_LOOP_ABLATION = args.closed_loop_ablation
    config.USE_PCSC = not args.disable_pcsc
    config.PCSC_DESIGN = args.pcsc_design
    config.USE_PPOT = not args.disable_ppot
    config.PROTOTYPE_ABLATION = args.prototype_ablation
    config.PPOT_COST_ABLATION = args.ppot_cost_ablation

    cost_preset = get_ppot_cost_component_preset(args.ppot_cost_ablation)
    config.OT_LAMBDA_PROB = cost_preset.ot_lambda_prob
    config.OT_LAMBDA_BOUNDARY = cost_preset.ot_lambda_boundary
    config.OT_LAMBDA_FEAT = cost_preset.ot_lambda_feat
    config.OT_LAMBDA_EDGE = cost_preset.ot_lambda_edge
    config.OT_USE_LOW_LEVEL_EDGE = cost_preset.use_low_level_edge
    config.OT_USE_BG_EXCLUSION = cost_preset.use_bg_exclusion


def main():
    args = parse_args()

    if args.dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choices: {VALID_DATASETS}")
    if (args.use_sam_prior or args.model in {'point2mask_v4_sam', 'point2mask_v4_synfoc'}) and not args.sam_prior_dir:
        raise ValueError(f"--model {args.model} requires --sam_prior_dir")

    actual_save_dir = os.path.join(
        args.save_dir,
        args.model,
        args.dataset,
        args.closed_loop_ablation,
        args.prototype_ablation,
        args.ppot_cost_ablation,
    )
    if args.disable_pcsc:
        actual_save_dir = os.path.join(actual_save_dir, 'no_pcsc')
    elif args.pcsc_design != 'full':
        actual_save_dir = os.path.join(actual_save_dir, f'pcsc_{args.pcsc_design}')
    if args.disable_ppot:
        actual_save_dir = os.path.join(actual_save_dir, 'no_ppot')
    sinkhorn_tag = _sinkhorn_iter_tag(args.ot_iters)
    if sinkhorn_tag is not None:
        actual_save_dir = os.path.join(actual_save_dir, sinkhorn_tag)
    geodesic_tag = _geodesic_cost_tag(args)
    if geodesic_tag is not None:
        actual_save_dir = os.path.join(actual_save_dir, geodesic_tag)
    os.makedirs(actual_save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Closed-loop ablation: {args.closed_loop_ablation}")
    print(f"PCSC: {'disabled' if args.disable_pcsc else 'enabled'}")
    if not args.disable_pcsc:
        print(f"PCSC design: {args.pcsc_design}")
    print(f"PPOT: {'disabled' if args.disable_ppot else 'enabled'}")
    print(f"Prototype ablation: {args.prototype_ablation}")
    print(f"PPOT cost ablation: {args.ppot_cost_ablation}")
    print(f"Sinkhorn iterations: {args.ot_iters}")
    print(f"Spatial distance mode: {args.ot_spatial_distance_mode}")
    if args.ot_spatial_distance_mode == 'dijkstra':
        print(f"Geodesic neighborhood: {args.ot_geodesic_neighborhood}")
        print(f"Path cost mode: {args.ot_path_cost_mode}")
        print(f"Boundary barrier: {'disabled' if args.ot_disable_boundary_barrier else 'enabled'}")
    print(f"SAM prior: {'enabled' if (args.use_sam_prior or args.sam_prior_dir) else 'disabled'}")
    if args.sam_prior_dir:
        print(f"SAM prior dir: {args.sam_prior_dir}")
    if args.model == 'point2mask_v4_synfoc':
        print(f"SynFoC correction weight: {args.synfoc_weight}")
    print(f"Save dir: {actual_save_dir}")

    active_config = Point2MaskV4Config
    apply_overrides(active_config, args)

    train_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split_dir=args.split_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_fg_points=args.num_fg_points,
        num_bg_points=args.num_bg_points,
        min_point_distance=args.min_point_distance,
        point_radius=args.point_radius,
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        test_noise_std=args.test_noise_std,
        sam_prior_dir=args.sam_prior_dir if (args.use_sam_prior or args.sam_prior_dir or args.model in {'point2mask_v4_sam', 'point2mask_v4_synfoc'}) else None,
        sam_prior_required=args.sam_prior_required,
        regenerate_splits=args.regenerate_splits,
    )

    if args.mode == 'test' and args.test_full_dataset:
        test_loader.dataset.samples = discover_samples(args.data_root, args.dataset)

    if args.mode == 'test' and args.test_sample_name:
        needle = args.test_sample_name
        filtered = []
        for sample in test_loader.dataset.samples:
            name = sample['name']
            stem = os.path.splitext(name)[0]
            if name == needle or stem == needle:
                filtered.append(sample)
        if not filtered:
            raise ValueError(f"No test sample matched '{needle}' in dataset '{args.dataset}'.")
        test_loader.dataset.samples = filtered

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model = build_model(active_config, device)
    print_model_summary(model, args.model)

    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(
        optimizer,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
        use_cosine=getattr(active_config, 'USE_COSINE_ANNEALING', True),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=active_config,
        dataset_name=args.dataset,
        model_name=args.model,
        save_dir=actual_save_dir,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_experiment=args.swanlab_experiment,
    )

    if args.mode == 'train':
        print(f"\n{'='*60}")
        print(f"Training on {args.dataset} for {args.epochs} epochs")
        print(f"{'='*60}")

        trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            save_test_visuals=args.save_val_visuals,
            num_test_visuals=args.num_val_visuals,
        )
        eval_seeds = args.eval_seeds or list(range(args.seed, args.seed + 5))
        trainer.test_multiple_seeds(
            test_loader,
            seeds=eval_seeds,
            output_dir=os.path.join(actual_save_dir, 'test_predictions', 'multi_seed'),
        )
        print(f"\nTest predictions saved under: {os.path.join(actual_save_dir, 'test_predictions')}")
        print(f"Latest checkpoint saved to: {os.path.join(actual_save_dir, 'checkpoints', 'latest.pth')}")

    elif args.mode == 'test':
        if args.checkpoint is None:
            args.checkpoint = os.path.join(actual_save_dir, 'checkpoints', 'latest.pth')

        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                f"Train the model first or specify a valid checkpoint with --checkpoint"
            )

        print(f"\n{'='*60}")
        print(f"Testing on {args.dataset}")
        print(f"Checkpoint: {args.checkpoint}")
        if args.test_noise_std > 0:
            print(f"Test noise std: {args.test_noise_std}")
        print(f"{'='*60}")

        trainer.load_checkpoint(args.checkpoint)
        output_dir = None
        if args.test_noise_std > 0:
            noise_tag = str(args.test_noise_std).replace('.', 'p')
            output_dir = os.path.join(actual_save_dir, 'test_predictions', f'gaussian_noise_{noise_tag}')
        eval_seeds = args.eval_seeds or list(range(args.seed, args.seed + 5))
        final_output_dir = output_dir or os.path.join(actual_save_dir, 'test_predictions', 'multi_seed')
        trainer.test_multiple_seeds(
            test_loader,
            seeds=eval_seeds,
            output_dir=final_output_dir,
        )
        print(f"Test predictions saved to: {final_output_dir}")

    return 0


if __name__ == '__main__':
    main()
