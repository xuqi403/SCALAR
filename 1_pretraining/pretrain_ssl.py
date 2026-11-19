#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint for SCALAR self-supervised pretraining.

All training and ROI operations live in ``scalar_ssl/``. This compatibility
module preserves the previous imported symbol names without duplicating code.
"""
from __future__ import annotations
from scalar_ssl.settings import *
from scalar_ssl import settings
from scalar_ssl.utils import (
    seed_everything,
    seed_worker,
    progress,
    clean_text,
    safe_stem,
    video_uid,
    write_image,
    read_image,
    resolve_video_path,
    image_tensor,
    denormalize_tensor,
    laplacian_blur_score,
    letterbox_bgr,
    load_module
)
from scalar_ssl.roi import (
    load_training_roi_prior,
    canonical_box,
    choose_video_crop_size,
    uniform_candidate_indices,
    read_selected_frames,
    choose_detection,
    _copy_box,
    box_area,
    box_center,
    box_wh,
    box_iou,
    valid_yolo_candidates,
    box_motion_metrics,
    is_temporally_consistent_box,
    predict_track_center,
    trajectory_error_box_diag,
    smooth_roi_box,
    select_tracked_box,
    best_neighbor_support,
    select_confirmed_center_anchor,
    propagated_fallback_box,
    bidirectional_local_track,
    add_local_geometry_change,
    greedy_diverse_indices
)
from scalar_ssl.cache import (
    pick_patient_id,
    clip_path_columns,
    clip_index_columns,
    _cache_row_files_exist,
    save_tracking_qc_preview,
    build_roi_cache
)
from scalar_ssl.dataset import (
    _sample_spatial_params,
    _apply_spatial,
    _photo_aug,
    make_ssl_clip_views,
    sample_context_mask,
    CarotidSSLDataset
)
from scalar_ssl.model import (
    load_dinov3,
    group_norm,
    DenseProjector,
    DensePredictor,
    CrossSectionContextHead,
    SSLStudent,
    SSLTeacher,
    set_student_trainable,
    apply_student_train_mode
)
from scalar_ssl.losses import (
    negative_cosine_dense_per_sample,
    negative_cosine_vector_per_sample,
    masked_mean,
    frame_ssl_loss,
    context_prediction_loss,
    context_lambda,
    ema_momentum,
    update_teacher
)
from scalar_ssl.trainer import (
    lr_multiplier,
    split_ssl_manifest,
    optimizer_groups,
    run_ssl_epoch,
    cpu_state_dict
)
from scalar_ssl.visualization import (
    select_probe_df,
    extract_backbone_features,
    effective_rank,
    cosine_rows,
    feature_heatmap,
    put_label,
    save_feature_map_panel,
    nearest_neighbor_indices,
    save_neighbor_panel,
    save_training_plot,
    raw_clip_tensor_from_row,
    save_context_attention_artifacts,
    build_probe_artifacts_before_training,
    finalize_probe_artifacts
)
from scalar_ssl.checkpoints import (
    safe_torch_load,
    state_dict_overlap,
    save_backbone_in_source_format,
    verify_hub_weight_file
)
from scalar_ssl.runner import (
    parse_args,
    main
)

if __name__ == "__main__":
    main()
