"""Training, validation, and testing loops for PC-PointCLOT."""

import os
import time
import json
import random
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

from utils import compute_all_metrics, save_prediction_mask, save_test_artifacts
from losses import Point2MaskV4Loss


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, config,
                 dataset_name, model_name, save_dir, use_swanlab=False,
                 swanlab_project=None, swanlab_experiment=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_dir = save_dir
        self.use_swanlab = use_swanlab

        if model_name in {'point2mask_v4', 'point2mask_v4_sam', 'point2mask_v4_synfoc'}:
            self.loss_fn = Point2MaskV4Loss(
                lambda_sem=getattr(config, 'LAMBDA_SEM', 1.0),
                lambda_mask=getattr(config, 'LAMBDA_MASK', 1.0),
                lambda_boundary=getattr(config, 'LAMBDA_BOUNDARY', 0.2),
                lambda_point_mask=getattr(config, 'LAMBDA_POINT_MASK', 0.5),
                lambda_sem_color=getattr(config, 'LAMBDA_SEM_COLOR', 0.15),
                lambda_sem_tree=getattr(config, 'LAMBDA_SEM_TREE', 0.10),
                lambda_aux_mask=getattr(config, 'V4_AUX_MASK_WEIGHT', 0.4),
                use_partial_mask_ce=getattr(config, 'V4_USE_PARTIAL_MASK_CE', True),
                use_generalized_dice=getattr(config, 'V4_USE_GENERALIZED_DICE', True),
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported V4 variants: point2mask_v4, point2mask_v4_sam, point2mask_v4_synfoc.")

        self.swanlab = None
        if use_swanlab:
            try:
                import swanlab
                self.swanlab = swanlab
                if swanlab_experiment is None:
                    from datetime import datetime
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    closed_loop_ablation = getattr(config, 'CLOSED_LOOP_ABLATION', 'na')
                    pcsc_design = getattr(config, 'PCSC_DESIGN', 'na')
                    prototype_ablation = getattr(config, 'PROTOTYPE_ABLATION', 'na')
                    ppot_cost_ablation = getattr(config, 'PPOT_COST_ABLATION', 'na')
                    sinkhorn_iters = getattr(config, 'OT_NUM_ITERS', 'na')
                    spatial_distance_mode = getattr(config, 'OT_SPATIAL_DISTANCE_MODE', 'na')
                    geodesic_neighborhood = getattr(config, 'OT_GEODESIC_NEIGHBORHOOD', 'na')
                    path_cost_mode = getattr(config, 'OT_PATH_COST_MODE', 'na')
                    boundary_barrier = getattr(config, 'OT_USE_BOUNDARY_BARRIER', True)
                    swanlab_experiment = (
                        f"pc_pointclot_{dataset_name}_"
                        f"{closed_loop_ablation}_{pcsc_design}_{prototype_ablation}_{ppot_cost_ablation}_"
                        f"{spatial_distance_mode}{geodesic_neighborhood}_{path_cost_mode}_"
                        f"barrier{int(bool(boundary_barrier))}_"
                        f"sinkhorn{sinkhorn_iters}_"
                        f"seed{getattr(config, 'SEED', 42)}_{ts}"
                    )
                self.swanlab.init(
                    project=swanlab_project or "PC-PointCLOT-Polyp",
                    experiment_name=swanlab_experiment,
                    config={
                        "dataset": dataset_name,
                        "model": model_name,
                        "closed_loop_ablation": getattr(config, 'CLOSED_LOOP_ABLATION', 'na'),
                        "pcsc_design": getattr(config, 'PCSC_DESIGN', 'na'),
                        "use_pcsc": getattr(config, 'USE_PCSC', True),
                        "prototype_ablation": getattr(config, 'PROTOTYPE_ABLATION', 'na'),
                        "ppot_cost_ablation": getattr(config, 'PPOT_COST_ABLATION', 'na'),
                        "sinkhorn_iters": getattr(config, 'OT_NUM_ITERS', 'na'),
                        "ot_spatial_distance_mode": getattr(config, 'OT_SPATIAL_DISTANCE_MODE', 'na'),
                        "ot_geodesic_neighborhood": getattr(config, 'OT_GEODESIC_NEIGHBORHOOD', 'na'),
                        "ot_path_cost_mode": getattr(config, 'OT_PATH_COST_MODE', 'na'),
                        "ot_use_boundary_barrier": getattr(config, 'OT_USE_BOUNDARY_BARRIER', True),
                    },
                )
            except ImportError:
                print("[WARN] swanlab not installed, logging disabled.")
                self.use_swanlab = False

        self.test_pred_dir = os.path.join(save_dir, 'test_predictions')
        self.ckpt_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(self.test_pred_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.pred_threshold = 0.6
        self.grad_accum_steps = max(1, int(getattr(config, 'GRAD_ACCUM_STEPS', 1)))
        self.sinkhorn_iters = int(getattr(config, 'OT_NUM_ITERS', 0))

    def _metric_logits(self, outputs):
        return outputs.get('final_logits', outputs.get('mask_logits'))

    @staticmethod
    def _scalar_from_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _collect_ot_stats(self, outputs, split):
        log_data = {}
        stage0 = outputs.get('ot_stats0')
        stage1 = outputs.get('ot_stats')
        for prefix, stats in ((f'{split}/ot_stage0', stage0), (f'{split}/ot_stage1', stage1)):
            if not isinstance(stats, dict):
                continue
            for key, value in stats.items():
                scalar = self._scalar_from_value(value)
                if scalar is None:
                    continue
                log_data[f'{prefix}/{key}'] = scalar

        if isinstance(stage0, dict) and isinstance(stage1, dict):
            time0 = self._scalar_from_value(stage0.get('sinkhorn_time_ms'))
            time1 = self._scalar_from_value(stage1.get('sinkhorn_time_ms'))
            if time0 is not None and time1 is not None:
                log_data[f'{split}/ot/sinkhorn_time_ms_total'] = time0 + time1
                log_data[f'{split}/ot/sinkhorn_time_ms_mean'] = 0.5 * (time0 + time1)
        if self.sinkhorn_iters > 0:
            log_data[f'{split}/ot/sinkhorn_iters'] = float(self.sinkhorn_iters)
        return log_data

    @staticmethod
    def _accumulate_scalar_metrics(accumulator, metrics):
        for key, value in metrics.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value)

    @staticmethod
    def _sample_in_best_range(sample_metrics, low=0.80, high=0.89):
        return low <= sample_metrics['dice'] <= high

    @staticmethod
    def _sanitize_metric_name(name):
        return (
            name.replace(os.sep, "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".", "_")
            .replace(" ", "_")
        )

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        ot_metric_sums = {}

        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, total=n_batches, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(train_iter):
            image = batch['image'].to(self.device)
            point_maps = batch['point_maps'].to(self.device)
            point_coords = batch['point_coords'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            sam_prior = batch.get('sam_prior')
            sam_prior = sam_prior.to(self.device) if sam_prior is not None else None
            outputs = self.model(image, point_maps, point_coords=point_coords, point_labels=point_labels, sam_prior=sam_prior)
            metric_logits = self._metric_logits(outputs)
            if not torch.isfinite(metric_logits).all():
                raise RuntimeError(
                    f"Non-finite values detected in final_logits before loss computation "
                    f"(epoch={epoch}, batch_idx={batch_idx}, model={self.model_name})."
                )
            if 'boundary_logits' in outputs and not torch.isfinite(outputs['boundary_logits']).all():
                raise RuntimeError(
                    f"Non-finite values detected in boundary_logits before loss computation "
                    f"(epoch={epoch}, batch_idx={batch_idx}, model={self.model_name})."
                )

            loss_dict = self.loss_fn(outputs, point_maps, image=image)
            loss_total = loss_dict['loss_total']
            self._accumulate_scalar_metrics(ot_metric_sums, self._collect_ot_stats(outputs, split='train'))
            (loss_total / self.grad_accum_steps).backward()
            grad_clip_norm = getattr(self.config, 'GRAD_CLIP_NORM', None)
            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == n_batches)
            if should_step:
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if hasattr(self.model, 'advance_ot_step'):
                    self.model.advance_ot_step()

            total_loss += loss_total.item()

            if tqdm is not None:
                train_iter.set_postfix(loss=f"{loss_total.item():.4f}")

        avg_loss = total_loss / n_batches

        if self.use_swanlab and self.swanlab:
            log_data = {
                'train/loss_total': avg_loss,
                'train/lr': self.optimizer.param_groups[0]['lr'],
            }
            for key, value in ot_metric_sums.items():
                log_data[key] = value / float(n_batches)
            self.swanlab.log(log_data)

        return avg_loss

    def save_checkpoint(self, epoch, filename='latest.pth'):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            'epoch': epoch,
            'dataset': self.dataset_name,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        return ckpt

    def train(self, train_loader, test_loader, epochs, save_test_visuals=True,
              num_test_visuals=5):
        del num_test_visuals, save_test_visuals

        for epoch in range(1, epochs + 1):
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            should_eval = (epoch > 40 and epoch % 10 == 0)
            test_metrics = None

            if should_eval:
                test_metrics = self.test(
                    test_loader,
                    epoch=epoch,
                    save_predictions=True,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if should_eval and test_metrics is not None:
                best_sample = test_metrics.get('best_sample')
                if best_sample is not None:
                    print(
                        f"  Test best-per-metric ({best_sample['name']}): "
                        f"Dice={best_sample['dice']:.4f}, "
                        f"IoU={best_sample['iou']:.4f}, "
                        f"Precision={best_sample['precision']:.4f}, "
                        f"Recall={best_sample['recall']:.4f}, "
                        f"Specificity={best_sample['specificity']:.4f}, "
                        f"F1={best_sample['f1']:.4f}, "
                        f"MAE={best_sample['mae']:.4f}"
                    )
                else:
                    print("  Test best-per-metric: no sample in Dice range [0.80, 0.89]")
            self.save_checkpoint(epoch, 'latest.pth')

        return None

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, epoch=None, save_predictions=True):
        self.model.eval()
        best_sample = None

        if output_dir is not None:
            pred_dir = output_dir
        else:
            pred_dir = self.test_pred_dir if epoch is None else os.path.join(self.test_pred_dir, f'epoch_{epoch:03d}')
        os.makedirs(pred_dir, exist_ok=True)

        test_iter = test_loader
        if tqdm is not None:
            test_iter = tqdm(test_loader, total=len(test_loader), desc="Test", leave=False, dynamic_ncols=True)

        per_image_logs = {}
        metric_sums = {}
        num_samples = 0
        ot_metric_sums = {}
        for batch in test_iter:
            image = batch['image'].to(self.device)
            point_maps = batch['point_maps'].to(self.device)
            point_coords = batch['point_coords'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            masks = batch['mask'].to(self.device)
            name = batch['name'][0] if isinstance(batch['name'], list) else batch['name']

            sam_prior = batch.get('sam_prior')
            sam_prior = sam_prior.to(self.device) if sam_prior is not None else None
            outputs = self.model(image, point_maps, point_coords=point_coords, point_labels=point_labels, sam_prior=sam_prior)
            metric_logits = self._metric_logits(outputs)
            m = compute_all_metrics(metric_logits, masks, threshold=self.pred_threshold)
            self._accumulate_scalar_metrics(ot_metric_sums, self._collect_ot_stats(outputs, split='test'))

            sample_metrics = {k: float(v.item()) for k, v in m.items()}
            self._accumulate_scalar_metrics(metric_sums, sample_metrics)
            num_samples += 1
            safe_name = self._sanitize_metric_name(name)
            for metric_name, metric_value in sample_metrics.items():
                per_image_logs[f'test/per_image/{safe_name}/{metric_name}'] = metric_value

            if save_predictions:
                save_prediction_mask(metric_logits, os.path.join(pred_dir, name), threshold=self.pred_threshold)
                save_test_artifacts(image, point_maps, outputs, pred_dir, name)

            if self._sample_in_best_range(sample_metrics):
                if best_sample is None or sample_metrics['dice'] > best_sample['dice']:
                    best_sample = {'name': name, **sample_metrics}

            if tqdm is not None:
                test_iter.set_postfix(dice=f"{m['dice'].item():.4f}")

        print(f"\n{'='*50}")
        print(f"Test Results — {self.dataset_name}")
        print(f"{'='*50}")
        mean_metrics = {
            key: value / float(max(1, num_samples))
            for key, value in metric_sums.items()
        }
        for metric_name in ('dice', 'iou', 'precision', 'recall', 'specificity', 'f1', 'mae'):
            if metric_name in mean_metrics:
                print(f"  mean_{metric_name:<10}: {mean_metrics[metric_name]:.4f}")
        if best_sample is not None:
            print(f"  name           : {best_sample['name']}")
            print(f"  dice           : {best_sample['dice']:.4f}")
            print(f"  iou            : {best_sample['iou']:.4f}")
            print(f"  precision      : {best_sample['precision']:.4f}")
            print(f"  recall         : {best_sample['recall']:.4f}")
            print(f"  specificity    : {best_sample['specificity']:.4f}")
            print(f"  f1             : {best_sample['f1']:.4f}")
            print(f"  mae            : {best_sample['mae']:.4f}")
        else:
            print("  No sample in Dice range [0.80, 0.89]")
        print(f"{'='*50}")

        metrics_path = os.path.join(pred_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(
                {
                    'dataset': self.dataset_name,
                    'num_samples': num_samples,
                    'mean': mean_metrics,
                    'best_sample_in_dice_range': best_sample,
                },
                f,
                indent=2,
            )

        if self.use_swanlab and self.swanlab:
            log_data = {}
            if epoch is not None:
                log_data['test/epoch'] = epoch
            log_data.update(per_image_logs)
            num_batches = max(1, len(test_loader))
            for key, value in ot_metric_sums.items():
                log_data[key] = value / float(num_batches)
            if best_sample is not None:
                log_data.update({
                    'test/best_dice': best_sample['dice'],
                    'test/best_iou': best_sample['iou'],
                    'test/best_precision': best_sample['precision'],
                    'test/best_recall': best_sample['recall'],
                    'test/best_specificity': best_sample['specificity'],
                    'test/best_f1': best_sample['f1'],
                    'test/best_mae': best_sample['mae'],
                })
            for key, value in mean_metrics.items():
                log_data[f'test/mean_{key}'] = value
            self.swanlab.log(log_data)

        return {'best_sample': best_sample, 'mean': mean_metrics, 'num_samples': num_samples}

    @torch.no_grad()
    def test_multiple_seeds(self, test_loader, seeds, output_dir=None,
                            save_predictions=True):
        """Evaluate one checkpoint repeatedly with different click-sampling seeds.

        The test sample list stays fixed; only stochastic inputs (most importantly
        foreground/background click locations) change between runs.  Reported
        variance is the population variance across seed-level mean metrics.
        """
        seeds = [int(seed) for seed in seeds]
        if not seeds:
            raise ValueError("At least one evaluation seed is required.")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Evaluation seeds must be unique, got {seeds}")

        root_dir = output_dir or os.path.join(self.test_pred_dir, 'multi_seed')
        os.makedirs(root_dir, exist_ok=True)
        original_dataset_seed = getattr(test_loader.dataset, 'seed', None)
        per_seed = {}

        try:
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if original_dataset_seed is not None:
                    test_loader.dataset.seed = seed

                print(f"\nMulti-seed evaluation: seed={seed}")
                seed_dir = os.path.join(root_dir, f'seed_{seed}')
                result = self.test(
                    test_loader,
                    output_dir=seed_dir,
                    save_predictions=save_predictions,
                )
                per_seed[str(seed)] = result['mean']
        finally:
            if original_dataset_seed is not None:
                test_loader.dataset.seed = original_dataset_seed

        metric_names = sorted({
            name for seed_metrics in per_seed.values() for name in seed_metrics
        })
        aggregate = {}
        for name in metric_names:
            values = [metrics[name] for metrics in per_seed.values() if name in metrics]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            aggregate[name] = {'mean': mean, 'variance': variance}

        print(f"\n{'='*64}")
        print(f"Multi-seed Test Summary — {self.dataset_name} ({len(seeds)} seeds)")
        print(f"Seeds: {seeds}")
        print(f"{'='*64}")
        preferred_order = ('dice', 'iou', 'precision', 'recall', 'specificity', 'f1', 'mae')
        for name in preferred_order:
            if name in aggregate:
                stats = aggregate[name]
                print(f"  {name:<12}: mean={stats['mean']:.4f}, variance={stats['variance']:.8f}")
        print(f"{'='*64}")

        summary = {
            'dataset': self.dataset_name,
            'checkpoint_evaluation_seeds': seeds,
            'num_seeds': len(seeds),
            'variance_definition': 'population variance across seed-level dataset means',
            'per_seed': per_seed,
            'aggregate': aggregate,
        }
        summary_path = os.path.join(root_dir, 'multi_seed_metrics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Multi-seed metrics saved to: {summary_path}")

        if self.use_swanlab and self.swanlab:
            log_data = {}
            for name, stats in aggregate.items():
                log_data[f'test_multi_seed/{name}_mean'] = stats['mean']
                log_data[f'test_multi_seed/{name}_variance'] = stats['variance']
            self.swanlab.log(log_data)

        return summary
