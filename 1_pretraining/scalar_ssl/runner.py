"""CLI configuration and end-to-end SSL orchestration (methodology preserved)."""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import settings
from .settings import *
from . import cache as _ssl_mod_cache
from . import checkpoints as _ssl_mod_checkpoints
from . import dataset as _ssl_mod_dataset
from . import model as _ssl_mod_model
from . import trainer as _ssl_mod_trainer
from . import utils as _ssl_mod_utils
from . import visualization as _ssl_mod_visualization
from cohort import load_approved_manifest, write_private_resolved_manifest


def parse_args():
    p = argparse.ArgumentParser(description='SCALAR internal-only DINOv3 sweep-context SSL (5288-video cohort).')
    p.add_argument('--mode', choices=['all', 'prepare', 'train'], default='all')
    p.add_argument('--cohort_manifest', type=Path, required=True, help='PRIVATE internal-only CSV manifest; video_path column; 5288 unique videos.')
    p.add_argument('--video_root', type=Path, default=None, help='Root for relative video paths; default is CSV parent directory.')
    p.add_argument('--excluded_manifest', type=Path, action='append', default=[], help='Optional PRIVATE CSV(s) with excluded patient/video records (repeatable).')
    p.add_argument('--video_train_file', type=Path, default=None, help='PRIVATE original train_v5.py defining import_yolo_components/yolo_detect_boxes.')
    p.add_argument('--v5_cache_manifest', type=Path, default=None, help='PRIVATE original image-training ROI cache manifest (used for ROI-size prior).')
    p.add_argument('--dino_repo', type=Path, default=None, help='Locally cloned official Meta DINOv3 repository (must contain hubconf.py).')
    p.add_argument('--dino_weights', type=Path, default=None, help='Locally obtained original DINOv3 ViT-B/16 checkpoint.')
    p.add_argument('--output_dir', type=Path, default=PROJECT_ROOT / 'outputs' / 'pretraining')
    p.add_argument('--expected_videos', type=int, default=5288, help='Strict cohort size; 5288 for paper. Override only for local synthetic tests.')
    p.add_argument('--force_rebuild_cache', action='store_true')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    settings.EXPECTED_INTERNAL_VIDEO_COUNT = int(args.expected_videos)
    if settings.EXPECTED_INTERNAL_VIDEO_COUNT <= 0:
        raise ValueError('--expected_videos must be positive')
    settings.OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    settings.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    settings.ROI_CACHE_ROOT = settings.OUTPUT_ROOT / 'roi_cache'
    settings.CACHE_MANIFEST_OUT = settings.OUTPUT_ROOT / 'pretrain_roi_manifest.csv'
    settings.BACKBONE_OUT = settings.OUTPUT_ROOT / 'dinov3_vitb16_carotid_ssl.pth'
    settings.BACKBONE_RAW_OUT = settings.OUTPUT_ROOT / 'dinov3_vitb16_carotid_ssl_raw_state_dict.pth'
    settings.FULL_CHECKPOINT_OUT = settings.OUTPUT_ROOT / 'ssl_resume_checkpoint.pth'
    settings.VIDEO_TRAIN_FILE = (args.video_train_file or Path('__missing_roi_detector_adapter__.py')).expanduser().resolve()
    settings.V5_CACHE_MANIFEST = (args.v5_cache_manifest or Path('__optional_roi_prior_not_supplied__.csv')).expanduser().resolve()
    settings.DINO_REPO = (args.dino_repo or Path('__missing_dinov3_source__')).expanduser().resolve()
    settings.ORIGINAL_DINO_WEIGHTS = (args.dino_weights or Path('__missing_dinov3_weights__.pth')).expanduser().resolve()
    manifest_file = args.cohort_manifest.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve() if args.video_root else manifest_file.parent
    settings.VIDEO_ROOTS = [video_root]
    approved = load_approved_manifest(manifest_file, video_root=video_root, expected_videos=settings.EXPECTED_INTERNAL_VIDEO_COUNT, excluded_manifests=args.excluded_manifest)
    prepared_file = write_private_resolved_manifest(approved, settings.OUTPUT_ROOT / 'approved_internal_cohort_PRIVATE.csv')
    settings.VIDEO_TABLES = [prepared_file]
    settings.APPROVED_VIDEO_UIDS = {_ssl_mod_utils.video_uid(Path(row['video_path'])) for row in approved}
    if len(settings.APPROVED_VIDEO_UIDS) != settings.EXPECTED_INTERNAL_VIDEO_COUNT:
        raise RuntimeError('Video UID collision / duplicate in approved manifest')
    if args.mode in {'prepare', 'all'} and (not settings.VIDEO_TRAIN_FILE.is_file()):
        raise FileNotFoundError('Original ROI-detector adapter is required for cache preparation. Supply --video_train_file PATH_TO_train_v5.py (with its YOLO imports/weights).')
    if args.mode in {'train', 'all'} and (not (settings.DINO_REPO / 'hubconf.py').is_file() or not settings.ORIGINAL_DINO_WEIGHTS.is_file()):
        raise FileNotFoundError('Training requires the original DINOv3 repository and ViT-B/16 weights. Supply --dino_repo and --dino_weights.')
    if not settings.V5_CACHE_MANIFEST.is_file():
        print('WARNING: original V5 ROI-size prior manifest was not supplied; the original script will use its YOLO fallback. Provide --v5_cache_manifest to reproduce the original ROI policy exactly.')
    _ssl_mod_utils.seed_everything(args.seed)
    config = {'video_tables': [str(p) for p in settings.VIDEO_TABLES], 'video_train_file': str(settings.VIDEO_TRAIN_FILE), 'v5_cache_manifest': str(settings.V5_CACHE_MANIFEST), 'dino_repo': str(settings.DINO_REPO), 'original_dino_weights': str(settings.ORIGINAL_DINO_WEIGHTS), 'backbone_out': str(settings.BACKBONE_OUT), 'image_size': IMAGE_SIZE, 'ssl_layers': SSL_LAYERS, 'trainable_last_n_blocks': TRAINABLE_LAST_N_BLOCKS, 'cache_clips_per_video': CACHE_CLIPS_PER_VIDEO, 'clip_candidates_to_validate': CLIP_CANDIDATES_TO_VALIDATE, 'context_offsets': list(CONTEXT_OFFSETS), 'tracking_radius_frames': TRACK_RADIUS_FRAMES, 'cache_schema_version': CACHE_SCHEMA_VERSION, 'candidate_frames_per_video': DETECTION_CANDIDATES_PER_VIDEO, 'max_roi_scale_from_training_median': MAX_ROI_SCALE_FROM_TRAIN_MEDIAN, 'roi_smooth_alpha': ROI_SMOOTH_ALPHA, 'center_confirm_min_neighbors': CENTER_CONFIRM_MIN_NEIGHBORS, 'max_track_gap_frames': MAX_TRACK_GAP_FRAMES, 'min_temporal_iou': MIN_TEMPORAL_IOU, 'max_center_shift_box_diag_ratio': MAX_CENTER_SHIFT_BOX_DIAG_RATIO, 'max_center_shift_image_diag_ratio': MAX_CENTER_SHIFT_IMAGE_DIAG_RATIO, 'min_box_size_ratio': MIN_BOX_SIZE_RATIO, 'max_box_size_ratio': MAX_BOX_SIZE_RATIO, 'track_iou_weight': TRACK_IOU_WEIGHT, 'track_center_penalty': TRACK_CENTER_PENALTY, 'track_trajectory_penalty': TRACK_TRAJECTORY_PENALTY, 'track_size_penalty': TRACK_SIZE_PENALTY, 'center_coarse_prior_penalty': CENTER_COARSE_PRIOR_PENALTY, 'max_trajectory_error_box_diag_ratio': MAX_TRAJECTORY_ERROR_BOX_DIAG_RATIO, 'min_tracked_context_for_context_loss': MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS, 'epochs': args.epochs, 'batch_size': args.batch_size, 'num_workers': args.num_workers, 'lr_backbone': LR_BACKBONE, 'lr_head': LR_HEAD, 'weight_decay': WEIGHT_DECAY, 'ema_start': EMA_MOMENTUM_START, 'ema_end': EMA_MOMENTUM_END, 'lambda_dense': LAMBDA_DENSE, 'lambda_global': LAMBDA_GLOBAL, 'lambda_context_max': LAMBDA_CONTEXT_MAX, 'context_delay_fraction': CONTEXT_DELAY_FRACTION, 'context_ramp_fraction': CONTEXT_RAMP_FRACTION, 'context_frame_dropout_prob': CONTEXT_FRAME_DROPOUT_PROB, 'ssl_val_video_fraction': SSL_VAL_VIDEO_FRACTION, 'seed': args.seed}
    with open(settings.OUTPUT_ROOT / 'ssl_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print('=' * 88)
    print('Carotid DINOv3 domain-adaptive self-supervised pretraining')
    print('=' * 88)
    print(f'Mode:       {args.mode}')
    print('Video tables:')
    for p in settings.VIDEO_TABLES:
        print(f'  {p}')
    print(f'Output:     {settings.OUTPUT_ROOT}')
    print(f'Output DINO:{settings.BACKBONE_OUT}')
    print('Design: confirmed-center bidirectional ROI tracking + last-4 DINO blocks + tracked 5-frame SSL + weak learned sweep-context prediction')
    if args.mode in {'all', 'prepare'}:
        manifest = _ssl_mod_cache.build_roi_cache(force=args.force_rebuild_cache)
    else:
        if not settings.CACHE_MANIFEST_OUT.is_file():
            raise FileNotFoundError(f'Missing ROI cache manifest: {settings.CACHE_MANIFEST_OUT}. Run --mode prepare/all first.')
        manifest = pd.read_csv(settings.CACHE_MANIFEST_OUT, low_memory=False)
        manifest = manifest[manifest.apply(_ssl_mod_cache._cache_row_files_exist, axis=1)].copy().reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError('No valid cached ROI clips were found for the approved cohort.')
    actual_cached_uids = set(manifest['video_uid'].astype(str))
    unexpected_uids = actual_cached_uids - settings.APPROVED_VIDEO_UIDS
    if unexpected_uids:
        raise RuntimeError(f'Cache contains {len(unexpected_uids)} videos outside the approved 5,288-video internal cohort. Refusing to train on a stale or mixed cache.')
    if args.mode == 'prepare':
        print('ROI cache prepared. Stop because --mode prepare was requested.')
        return
    if len(manifest) < 100:
        raise RuntimeError(f'Too few cached ROI clips for SSL: {len(manifest)}')
    train_df, val_df = _ssl_mod_trainer.split_ssl_manifest(manifest, args.seed)
    train_df.to_csv(settings.OUTPUT_ROOT / 'ssl_train_manifest.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(settings.OUTPUT_ROOT / 'ssl_monitor_manifest.csv', index=False, encoding='utf-8-sig')
    print(f"SSL clips: train={len(train_df):,} ({train_df['video_uid'].nunique():,} videos; {len(train_df) * len(CONTEXT_OFFSETS):,} frame-instances) | monitor={len(val_df):,} ({val_df['video_uid'].nunique():,} videos)")
    train_ds = _ssl_mod_dataset.CarotidSSLDataset(train_df, training=True)
    val_ds = _ssl_mod_dataset.CarotidSSLDataset(val_df, training=False)
    gen = torch.Generator().manual_seed(args.seed)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_ssl_mod_utils.seed_worker, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    base_backbone = _ssl_mod_model.load_dinov3(settings.ORIGINAL_DINO_WEIGHTS)
    student = _ssl_mod_model.SSLStudent(base_backbone).to(device)
    _ssl_mod_model.set_student_trainable(student)
    teacher = _ssl_mod_model.SSLTeacher(backbone=copy.deepcopy(student.backbone), projector=copy.deepcopy(student.projector)).to(device).eval()
    context_head = _ssl_mod_model.CrossSectionContextHead().to(device)
    trainable_backbone = sum((p.numel() for p in student.backbone.parameters() if p.requires_grad))
    total_backbone = sum((p.numel() for p in student.backbone.parameters()))
    print(f'Trainable backbone params: {trainable_backbone:,}/{total_backbone:,} ({100 * trainable_backbone / total_backbone:.1f}%)')
    probe_info = _ssl_mod_visualization.build_probe_artifacts_before_training(student.backbone, manifest, device)
    optimizer = torch.optim.AdamW(_ssl_mod_trainer.optimizer_groups(student, context_head), weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP and device.type == 'cuda')
    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    history = []
    best_val = np.inf
    best_epoch = 0
    best_student_state = None
    best_teacher_state = None
    best_context_state = None
    for epoch in range(1, args.epochs + 1):
        tr, global_step = _ssl_mod_trainer.run_ssl_epoch(student, teacher, context_head, train_loader, optimizer, scaler, device, epoch, args.epochs, global_step, total_steps, train=True)
        va, _ = _ssl_mod_trainer.run_ssl_epoch(student, teacher, context_head, val_loader, None, scaler, device, epoch, args.epochs, global_step, total_steps, train=False)
        row = {'epoch': epoch, 'train_total_loss': tr['total_loss'], 'train_frame_loss': tr['frame_loss'], 'train_dense_loss': tr['dense_loss'], 'train_global_loss': tr['global_loss'], 'train_context_loss': tr['context_loss'], 'train_context_lambda': tr['context_lambda'], 'val_total_loss': va['total_loss'], 'val_frame_loss': va['frame_loss'], 'val_dense_loss': va['dense_loss'], 'val_global_loss': va['global_loss'], 'val_context_loss': va['context_loss'], 'val_alpha_m4': va.get('alpha_-4', np.nan), 'val_alpha_m2': va.get('alpha_-2', np.nan), 'val_alpha_p2': va.get('alpha_+2', np.nan), 'val_alpha_p4': va.get('alpha_+4', np.nan), 'ema_momentum': tr['ema_momentum'], 'lr_backbone': optimizer.param_groups[0]['lr'], 'lr_head': optimizer.param_groups[1]['lr']}
        history.append(row)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(settings.OUTPUT_ROOT / 'ssl_history.csv', index=False, encoding='utf-8-sig')
        _ssl_mod_visualization.save_training_plot(hist_df, settings.OUTPUT_ROOT / 'visualization' / 'ssl_training_curves.png')
        print(f"Epoch {epoch:02d}/{args.epochs} | train {tr['total_loss']:.4f} | monitor {va['total_loss']:.4f} | frame {va['frame_loss']:.4f} | context {va['context_loss']:.4f}")
        if va['total_loss'] < best_val:
            best_val = float(va['total_loss'])
            best_epoch = epoch
            best_student_state = _ssl_mod_trainer.cpu_state_dict(student)
            best_teacher_state = _ssl_mod_trainer.cpu_state_dict(teacher)
            best_context_state = _ssl_mod_trainer.cpu_state_dict(context_head)
            torch.save({'student_state_dict': best_student_state, 'teacher_state_dict': best_teacher_state, 'context_head_state_dict': best_context_state, 'epoch': epoch, 'best_monitor_loss': best_val, 'global_step': global_step, 'config': config}, settings.FULL_CHECKPOINT_OUT)
    if best_student_state is None:
        raise RuntimeError('No SSL checkpoint was saved.')
    student.load_state_dict(best_student_state, strict=True)
    teacher.load_state_dict(best_teacher_state, strict=True)
    context_head.load_state_dict(best_context_state, strict=True)
    print(f'Best SSL monitor epoch: {best_epoch} | loss={best_val:.5f}')
    fmt = _ssl_mod_checkpoints.save_backbone_in_source_format(teacher.backbone, settings.ORIGINAL_DINO_WEIGHTS, settings.BACKBONE_OUT)
    print(f'Saved adapted EMA-teacher backbone: {settings.BACKBONE_OUT} | format={fmt}')
    print(f'Raw backbone state_dict: {settings.BACKBONE_RAW_OUT}')
    _ssl_mod_visualization.finalize_probe_artifacts(teacher.backbone, probe_info, device)
    _ssl_mod_visualization.save_context_attention_artifacts(student, teacher, context_head, val_df, device)
    student = student.cpu()
    teacher = teacher.cpu()
    context_head = context_head.cpu()
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ok, message = _ssl_mod_checkpoints.verify_hub_weight_file(settings.BACKBONE_OUT)
    if not ok:
        print(f'Compatibility check failed with source-format export: {message}')
        raw_state = _ssl_mod_checkpoints.safe_torch_load(settings.BACKBONE_RAW_OUT)
        torch.save(raw_state, settings.BACKBONE_OUT)
        ok, message = _ssl_mod_checkpoints.verify_hub_weight_file(settings.BACKBONE_OUT)
    pd.DataFrame([{'backbone_out': str(settings.BACKBONE_OUT), 'format_initial_export': fmt, 'torch_hub_compatible': int(ok), 'message': message, 'best_epoch': best_epoch, 'best_monitor_loss': best_val}]).to_csv(settings.OUTPUT_ROOT / 'backbone_compatibility_check.csv', index=False, encoding='utf-8-sig')
    print('-' * 88)
    print(f'torch.hub compatibility: {ok} | {message}')
    if not ok:
        raise RuntimeError('Adapted weight file could not be loaded by the same DINOv3 torch.hub interface. Do not use it downstream until the local checkpoint wrapper is inspected.')
    print('DONE.')
    print(f'Adapted DINOv3 backbone for downstream image training: {settings.BACKBONE_OUT}')
    print('Use the adapted weights with the separately released SCALAR image/video training modules.')
