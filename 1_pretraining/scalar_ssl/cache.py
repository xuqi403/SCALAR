"""Five-frame ROI clip construction, cached manifests and tracking QC (methodology preserved)."""
from __future__ import annotations
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import settings
from .settings import *
from . import roi as _ssl_mod_roi
from . import utils as _ssl_mod_utils
from . import visualization as _ssl_mod_visualization


def pick_patient_id(row: pd.Series) -> str:
    for c in ['patient_id', 'ID', 'id', 'patient', 'patientID', 'PatientID']:
        if c in row.index:
            v = _ssl_mod_utils.clean_text(row.get(c))
            if v:
                return v
    return ''


def clip_path_columns() -> List[str]:
    return [f'frame_{off:+d}_path' for off in CONTEXT_OFFSETS]


def clip_index_columns() -> List[str]:
    return [f'frame_{off:+d}_idx' for off in CONTEXT_OFFSETS]


def _cache_row_files_exist(row: pd.Series) -> bool:
    try:
        if int(row.get('cache_schema_version', -1)) != CACHE_SCHEMA_VERSION:
            return False
    except Exception:
        return False
    for c in clip_path_columns():
        p = _ssl_mod_utils.clean_text(row.get(c, ''))
        if not p or not Path(p).is_file():
            return False
    return True


def save_tracking_qc_preview(manifest: pd.DataFrame) -> None:
    if manifest.empty or TRACKING_QC_PREVIEW_CLIPS <= 0:
        return
    vis_dir = settings.OUTPUT_ROOT / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    x = manifest.copy()
    x['_ctx_valid'] = pd.to_numeric(x.get('context_tracking_valid_count', 0), errors='coerce').fillna(0)
    n = min(TRACKING_QC_PREVIEW_CLIPS, len(x))
    n_hard = min((n + 1) // 2, len(x))
    hard = x.sort_values(['_ctx_valid', 'center_yolo_confidence'], ascending=[True, True]).head(n_hard)
    rest = x.drop(index=hard.index)
    n_rest = n - len(hard)
    if n_rest > 0 and len(rest):
        rnd = rest.sample(min(n_rest, len(rest)), random_state=SEED + 171)
        show = pd.concat([hard, rnd], ignore_index=True)
    else:
        show = hard.reset_index(drop=True)
    panel_rows = []
    for _, row in show.iterrows():
        tiles = []
        for off, c in zip(CONTEXT_OFFSETS, clip_path_columns()):
            img = _ssl_mod_utils.read_image(str(row[c]))
            img = cv2.resize(img, (TRACKING_QC_THUMB_SIZE, TRACKING_QC_THUMB_SIZE))
            valid = int(row.get(f'frame_{off:+d}_tracking_valid', 0))
            src = _ssl_mod_utils.clean_text(row.get(f'frame_{off:+d}_tracking_source', ''))
            conf = pd.to_numeric(pd.Series([row.get(f'frame_{off:+d}_yolo_confidence', np.nan)]), errors='coerce').iloc[0]
            traj = pd.to_numeric(pd.Series([row.get(f'frame_{off:+d}_trajectory_error_box_diag', np.nan)]), errors='coerce').iloc[0]
            if off == 0:
                lab = f't | center confirmed | conf={conf:.2f}' if np.isfinite(conf) else 't | center confirmed'
            else:
                conf_s = f'{conf:.2f}' if np.isfinite(conf) else 'NA'
                traj_s = f'{traj:.2f}' if np.isfinite(traj) else 'NA'
                lab = f't{off:+d} | valid={valid} | {src} | c={conf_s} tr={traj_s}'
            tiles.append(_ssl_mod_visualization.put_label(img, lab))
        panel_rows.append(np.concatenate(tiles, axis=1))
    if panel_rows:
        _ssl_mod_utils.write_image(vis_dir / 'tracking_qc_preview.jpg', np.concatenate(panel_rows, axis=0), quality=94)


def build_roi_cache(force: bool=False) -> pd.DataFrame:
    settings.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    settings.ROI_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    table_parts = []
    for table_order, table_path in enumerate(settings.VIDEO_TABLES):
        table_path = Path(table_path)
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        df = pd.read_csv(table_path, low_memory=False) if table_path.suffix.lower() == '.csv' else pd.read_excel(table_path)
        if VIDEO_PATH_COLUMN not in df.columns:
            raise KeyError(f'Missing {VIDEO_PATH_COLUMN!r} in {table_path}')
        df = df.copy()
        df['_video_table_source'] = table_path.name
        df['_video_table_path'] = str(table_path)
        df['_video_table_row'] = np.arange(len(df), dtype=np.int64)
        df['_video_table_order'] = int(table_order)
        table_parts.append(df)
    if not table_parts:
        raise RuntimeError('VIDEO_TABLES is empty.')
    table = pd.concat(table_parts, ignore_index=True, sort=False)
    source_summary = table.groupby(['_video_table_order', '_video_table_source', '_video_table_path'], dropna=False).size().reset_index(name='table_rows').sort_values('_video_table_order')
    source_summary.to_csv(settings.OUTPUT_ROOT / 'video_table_input_summary.csv', index=False, encoding='utf-8-sig')
    print(f'Loaded {len(table):,} rows from {len(settings.VIDEO_TABLES)} video tables.')
    for _, sr in source_summary.iterrows():
        print(f"  {sr['_video_table_source']}: {int(sr['table_rows']):,} rows")
    rows = []
    seen = set()
    missing_rows = []
    duplicate_rows = []
    for _, row in table.iterrows():
        source = _ssl_mod_utils.clean_text(row.get('_video_table_source', ''))
        source_path = _ssl_mod_utils.clean_text(row.get('_video_table_path', ''))
        source_row = int(row.get('_video_table_row', -1))
        resolved = _ssl_mod_utils.resolve_video_path(row.get(VIDEO_PATH_COLUMN))
        if resolved is None:
            missing_rows.append({'video_table_source': source, 'video_table_path': source_path, 'video_table_row': source_row, 'stored_video_path': _ssl_mod_utils.clean_text(row.get(VIDEO_PATH_COLUMN))})
            continue
        key = str(resolved).lower()
        if key in seen:
            duplicate_rows.append({'video_table_source': source, 'video_table_path': source_path, 'video_table_row': source_row, 'stored_video_path': _ssl_mod_utils.clean_text(row.get(VIDEO_PATH_COLUMN)), 'resolved_video_path': str(resolved), 'reason': 'duplicate_resolved_video_path'})
            continue
        seen.add(key)
        rows.append({'video_path': str(resolved), 'patient_id': pick_patient_id(row), 'table_row': source_row, 'video_table_source': source, 'video_table_path': source_path})
    pd.DataFrame(missing_rows).to_csv(settings.OUTPUT_ROOT / 'video_paths_unresolved.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(duplicate_rows).to_csv(settings.OUTPUT_ROOT / 'video_table_duplicates_removed.csv', index=False, encoding='utf-8-sig')
    if not rows:
        raise RuntimeError('No usable videos resolved from VIDEO_TABLES.')
    if len(rows) != settings.EXPECTED_INTERNAL_VIDEO_COUNT:
        raise RuntimeError(f'Expected {settings.EXPECTED_INTERNAL_VIDEO_COUNT} approved internal videos, but prepared {len(rows)} after resolution/deduplication. Correct the private cohort manifest before training.')
    actual_uids = {_ssl_mod_utils.video_uid(Path(rec['video_path'])) for rec in rows}
    if actual_uids != settings.APPROVED_VIDEO_UIDS:
        raise RuntimeError('Prepared videos differ from the approved internal-only cohort.')
    resolved_df = pd.DataFrame(rows)
    resolved_df.to_csv(settings.OUTPUT_ROOT / 'resolved_video_cohort.csv', index=False, encoding='utf-8-sig')
    prior = _ssl_mod_roi.load_training_roi_prior(settings.V5_CACHE_MANIFEST)
    with open(settings.OUTPUT_ROOT / 'roi_prior.json', 'w', encoding='utf-8') as f:
        json.dump(prior, f, ensure_ascii=False, indent=2)
    print(f'Resolved unique videos: {len(rows):,}')
    print(f"Training ROI prior: {(prior if prior else 'NOT AVAILABLE -> YOLO fallback')}")
    train_lib = _ssl_mod_utils.load_module(settings.VIDEO_TRAIN_FILE, 'carotid_v5_train_for_ssl')
    YOLO, cvtColor, preprocess_input, resize_image = train_lib.import_yolo_components()
    yolo = YOLO()
    yolo.confidence = max(float(getattr(train_lib, 'YOLO_CONF_THRESHOLD', MIN_YOLO_CONFIDENCE)), MIN_YOLO_CONFIDENCE)
    failures = []
    tracking_audit_rows = []
    total_cached = 0
    for vi, rec in enumerate(_ssl_mod_utils.progress(rows, 'Prepare SSL sweep-clip cache'), start=1):
        vp = Path(rec['video_path'])
        uid = _ssl_mod_utils.video_uid(vp)
        vdir = settings.ROI_CACHE_ROOT / uid
        per_video_csv = vdir / 'roi_manifest.csv'
        if per_video_csv.is_file() and (not force):
            try:
                old = pd.read_csv(per_video_csv, low_memory=False)
                if len(old) and all((_cache_row_files_exist(r) for _, r in old.iterrows())):
                    old['video_table_source'] = rec['video_table_source']
                    old['video_table_path'] = rec['video_table_path']
                    old['video_table_row'] = rec['table_row']
                    old.to_csv(per_video_csv, index=False, encoding='utf-8-sig')
                    total_cached += len(old)
                    continue
            except Exception:
                pass
        try:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                raise RuntimeError('cannot_open')
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 2 * TRACK_RADIUS_FRAMES:
                raise RuntimeError('video_too_short_for_context')
            cand_idx = _ssl_mod_roi.uniform_candidate_indices(frame_count, DETECTION_CANDIDATES_PER_VIDEO)
            frames, w, h, fps, nframes = _ssl_mod_roi.read_selected_frames(vp, cand_idx)
            if not frames:
                raise RuntimeError('no_candidate_frames_decoded')
            detected: List[Dict[str, float]] = []
            prev_center = None
            for fi in sorted(frames):
                frame = frames[fi]
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes = train_lib.yolo_detect_boxes(yolo, pil, cvtColor, preprocess_input, resize_image)
                best = _ssl_mod_roi.choose_detection(boxes, prev_center, w, h)
                if best is None:
                    continue
                cx, cy = _ssl_mod_roi.box_center(best)
                prev_center = (cx, cy)
                blur = _ssl_mod_utils.laplacian_blur_score(frame)
                if blur < MIN_BLUR_SCORE:
                    continue
                detected.append({'frame_idx': int(fi), 'time_sec': float(fi / fps), 'time_fraction': float(fi / max(nframes - 1, 1)), 'yolo_confidence': float(best.get('score', np.nan)), 'box_w': float(best['x2'] - best['x1']), 'box_h': float(best['y2'] - best['y1']), 'center_x': cx, 'center_y': cy, 'center_x_norm': cx / max(w, 1), 'center_y_norm': cy / max(h, 1), 'blur_score': blur})
            if not detected:
                raise RuntimeError('no_valid_yolo_candidates')
            _ssl_mod_roi.add_local_geometry_change(detected)
            crop_w, crop_h, crop_source, crop_scale = _ssl_mod_roi.choose_video_crop_size(detected, w, h, prior)
            n_validate = min(max(CACHE_CLIPS_PER_VIDEO, CLIP_CANDIDATES_TO_VALIDATE), len(detected))
            keep = _ssl_mod_roi.greedy_diverse_indices(detected, n_validate)
            candidates = [detected[i] for i in keep]
            if not candidates:
                raise RuntimeError('no_clip_centers_selected')
            needed = sorted({int(d['frame_idx'] + off) for d in candidates for off in range(-TRACK_RADIUS_FRAMES, TRACK_RADIUS_FRAMES + 1) if 0 <= int(d['frame_idx'] + off) < nframes})
            clip_frames, w2, h2, fps2, nframes2 = _ssl_mod_roi.read_selected_frames(vp, needed)
            if w2 != w or h2 != h:
                raise RuntimeError('video_geometry_changed_between_passes')
            boxes_by_frame: Dict[int, List[Dict]] = {}
            for fi in needed:
                frame = clip_frames.get(fi)
                if frame is None:
                    boxes_by_frame[fi] = []
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes_by_frame[fi] = train_lib.yolo_detect_boxes(yolo, pil, cvtColor, preprocess_input, resize_image)
            validated = []
            for d in candidates:
                center_fi = int(d['frame_idx'])
                anchor, center_support, anchor_score, seed_neighbors = _ssl_mod_roi.select_confirmed_center_anchor(boxes_by_frame, center_fi, w, h, preferred_center=(float(d['center_x']), float(d['center_y'])), preferred_box_diag=math.hypot(float(d['box_w']), float(d['box_h'])))
                if anchor is None:
                    tracking_audit_rows.append({'video_uid': uid, 'video_path': str(vp), 'center_frame_idx': center_fi, 'video_table_source': rec['video_table_source'], 'video_table_row': rec['table_row'], 'accepted': 0, 'reason': 'center_not_locally_confirmed', 'context_tracking_valid_count': 0})
                    continue
                track = _ssl_mod_roi.bidirectional_local_track(boxes_by_frame, center_fi, anchor, seed_neighbors, w, h)
                ctx_valid = 0
                for off in CONTEXT_OFFSETS:
                    if off == 0:
                        continue
                    info = track.get(center_fi + int(off), {})
                    ctx_valid += int(info.get('valid', 0) == 1)
                eligible = int(ctx_valid >= MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS)
                validated.append({'candidate': d, 'anchor': anchor, 'center_support': int(center_support), 'anchor_score': float(anchor_score), 'track': track, 'context_valid_count': int(ctx_valid), 'context_loss_eligible': eligible})
            if not validated:
                raise RuntimeError('no_center_confirmed_local_tracks')
            validated.sort(key=lambda x: (int(x['context_loss_eligible']), int(x['context_valid_count']), int(x['center_support']), float(x['candidate'].get('yolo_confidence', 0.0))), reverse=True)
            selected_tracks = validated[:CACHE_CLIPS_PER_VIDEO]
            selected_tracks.sort(key=lambda x: int(x['candidate']['frame_idx']))
            vdir.mkdir(parents=True, exist_ok=True)
            out_rows = []
            for item in selected_tracks:
                d = item['candidate']
                center_fi = int(d['frame_idx'])
                anchor = item['anchor']
                track = item['track']
                clip_dir = vdir / f'clip_f{center_fi:06d}'
                clip_dir.mkdir(parents=True, exist_ok=True)
                center_blur = _ssl_mod_utils.laplacian_blur_score(clip_frames[center_fi]) if center_fi in clip_frames else float(d['blur_score'])
                row_out: Dict[str, Any] = {'cache_schema_version': CACHE_SCHEMA_VERSION, 'clip_uid': f'{uid}__f{center_fi:06d}', 'video_uid': uid, 'video_path': str(vp), 'patient_id': rec['patient_id'], 'video_table_source': rec['video_table_source'], 'video_table_path': rec['video_table_path'], 'video_table_row': rec['table_row'], 'center_frame_idx': center_fi, 'time_sec': float(center_fi / fps), 'time_fraction': float(center_fi / max(nframes - 1, 1)), 'video_width': w, 'video_height': h, 'video_fps': fps, 'video_frame_count': nframes, 'center_yolo_confidence': float(anchor.get('score', np.nan)), 'center_confirm_support': int(item['center_support']), 'center_anchor_score': float(item['anchor_score']), 'center_blur_score': float(center_blur), 'geometry_change': float(d.get('geometry_change', 0.0)), 'crop_width': crop_w, 'crop_height': crop_h, 'crop_source': crop_source, 'crop_scale_from_training_median': crop_scale, 'context_tracking_valid_count': int(item['context_valid_count']), 'context_loss_eligible': int(item['context_loss_eligible'])}
                ok_clip = True
                valid_conf = []
                for off in CONTEXT_OFFSETS:
                    fi = center_fi + int(off)
                    frame = clip_frames.get(fi)
                    if frame is None:
                        ok_clip = False
                        break
                    info = track.get(fi)
                    if info is None:
                        ok_clip = False
                        break
                    tracked_box = info['box']
                    cx, cy = _ssl_mod_roi.box_center(tracked_box)
                    x1, y1, x2, y2 = _ssl_mod_roi.canonical_box(cx, cy, crop_w, crop_h, w, h)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        ok_clip = False
                        break
                    roi448 = _ssl_mod_utils.letterbox_bgr(roi, IMAGE_SIZE)
                    cp = clip_dir / f'f{fi:06d}.jpg'
                    if not _ssl_mod_utils.write_image(cp, roi448):
                        ok_clip = False
                        break
                    raw_box = info.get('raw_box')
                    conf = float(raw_box.get('score', np.nan)) if raw_box is not None else np.nan
                    valid = int(info.get('valid', 0))
                    if valid and np.isfinite(conf):
                        valid_conf.append(conf)
                    m = info.get('metrics', {}) or {}
                    key = f'frame_{off:+d}'
                    row_out[f'{key}_idx'] = int(fi)
                    row_out[f'{key}_path'] = str(cp)
                    row_out[f'{key}_tracking_valid'] = valid
                    row_out[f'{key}_tracking_source'] = str(info.get('source', ''))
                    row_out[f'{key}_yolo_confidence'] = conf
                    row_out[f'{key}_temporal_iou'] = float(m.get('temporal_iou', np.nan))
                    row_out[f'{key}_center_shift_box_diag'] = float(m.get('center_shift_box_diag', np.nan))
                    row_out[f'{key}_center_shift_image_diag'] = float(m.get('center_shift_image_diag', np.nan))
                    row_out[f'{key}_box_size_ratio'] = float(m.get('box_size_ratio', np.nan))
                    row_out[f'{key}_trajectory_error_box_diag'] = float(m.get('trajectory_error_box_diag', np.nan))
                    row_out[f'{key}_crop_x1'] = x1
                    row_out[f'{key}_crop_y1'] = y1
                    row_out[f'{key}_crop_x2'] = x2
                    row_out[f'{key}_crop_y2'] = y2
                if not ok_clip:
                    continue
                row_out['cache_path'] = row_out['frame_+0_path']
                row_out['mean_valid_context_yolo_confidence'] = float(np.mean(valid_conf)) if valid_conf else np.nan
                out_rows.append(row_out)
                tracking_audit_rows.append({'video_uid': uid, 'video_path': str(vp), 'center_frame_idx': center_fi, 'video_table_source': rec['video_table_source'], 'video_table_row': rec['table_row'], 'accepted': 1, 'reason': 'accepted', 'context_tracking_valid_count': int(item['context_valid_count']), 'context_loss_eligible': int(item['context_loss_eligible']), 'center_confirm_support': int(item['center_support'])})
            if not out_rows:
                raise RuntimeError('no_tracked_clip_written')
            pd.DataFrame(out_rows).to_csv(per_video_csv, index=False, encoding='utf-8-sig')
            total_cached += len(out_rows)
        except Exception as e:
            failures.append({'video_path': str(vp), 'patient_id': rec['patient_id'], 'video_table_source': rec['video_table_source'], 'video_table_row': rec['table_row'], 'error': f'{type(e).__name__}: {e}'})
        if vi % 100 == 0:
            print(f'  videos {vi:,}/{len(rows):,} | cached clips so far {total_cached:,} | failures {len(failures):,}')
    del yolo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parts = []
    for rec in rows:
        uid = _ssl_mod_utils.video_uid(Path(rec['video_path']))
        pth = settings.ROI_CACHE_ROOT / uid / 'roi_manifest.csv'
        if not pth.is_file():
            continue
        try:
            x = pd.read_csv(pth, low_memory=False)
            if len(x):
                x['video_table_source'] = rec['video_table_source']
                x['video_table_path'] = rec['video_table_path']
                x['video_table_row'] = rec['table_row']
                parts.append(x)
        except Exception:
            continue
    if not parts:
        raise RuntimeError('ROI clip cache is empty for the currently configured VIDEO_TABLES.')
    manifest = pd.concat(parts, ignore_index=True, sort=False)
    manifest = manifest[manifest.apply(_cache_row_files_exist, axis=1)].copy()
    manifest = manifest.drop_duplicates(['clip_uid']).reset_index(drop=True)
    manifest.to_csv(settings.CACHE_MANIFEST_OUT, index=False, encoding='utf-8-sig')
    pd.DataFrame(failures).to_csv(settings.OUTPUT_ROOT / 'cache_failures.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(tracking_audit_rows).to_csv(settings.OUTPUT_ROOT / 'tracking_candidate_audit.csv', index=False, encoding='utf-8-sig')
    save_tracking_qc_preview(manifest)
    eligible_rate = float(pd.to_numeric(manifest.get('context_loss_eligible', 0), errors='coerce').mean()) if len(manifest) else np.nan
    summary = pd.DataFrame([{'cache_schema_version': CACHE_SCHEMA_VERSION, 'n_video_tables': len(settings.VIDEO_TABLES), 'table_rows': len(table), 'resolved_unique_videos': len(rows), 'duplicate_video_rows_removed': len(duplicate_rows), 'unresolved_video_rows': len(missing_rows), 'cached_videos': int(manifest['video_uid'].nunique()), 'cached_clips': len(manifest), 'cached_frame_instances': int(len(manifest) * len(CONTEXT_OFFSETS)), 'median_clips_per_video': float(manifest.groupby('video_uid').size().median()), 'mean_center_yolo_confidence': float(pd.to_numeric(manifest['center_yolo_confidence'], errors='coerce').mean()), 'mean_center_blur_score': float(pd.to_numeric(manifest['center_blur_score'], errors='coerce').mean()), 'mean_geometry_change': float(pd.to_numeric(manifest['geometry_change'], errors='coerce').mean()), 'mean_tracked_context_frames': float(pd.to_numeric(manifest['context_tracking_valid_count'], errors='coerce').mean()), 'context_loss_eligible_rate': eligible_rate, 'failures': len(failures)}])
    summary.to_csv(settings.OUTPUT_ROOT / 'cache_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return manifest
