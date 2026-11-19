"""Conservative YOLO detections and bidirectional sweep ROI tracking (methodology preserved)."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
from . import settings
from .settings import *


def load_training_roi_prior(cache_manifest: Path) -> Dict[str, float]:
    ratios_w, ratios_h = ([], [])
    cache_manifest = Path(cache_manifest)
    if not cache_manifest.is_file():
        return {}
    df = pd.read_csv(cache_manifest, low_memory=False)
    if 'split' in df.columns:
        df = df[df['split'].astype(str).str.lower().eq('train')]
    if len(df) > 1200:
        df = df.sample(1200, random_state=SEED)
    for row in df.itertuples(index=False):
        cp = Path(str(getattr(row, 'cache_path', '')))
        vw = float(getattr(row, 'video_width', np.nan))
        vh = float(getattr(row, 'video_height', np.nan))
        if not cp.is_file() or not (np.isfinite(vw) and np.isfinite(vh) and (vw > 0) and (vh > 0)):
            continue
        try:
            with np.load(cp, allow_pickle=False) as z:
                box = z['anchor_box'].astype(float)
            rw = (box[2] - box[0]) / vw
            rh = (box[3] - box[1]) / vh
            if np.isfinite(rw) and np.isfinite(rh) and (rw > 0) and (rh > 0):
                ratios_w.append(float(rw))
                ratios_h.append(float(rh))
        except Exception:
            continue
    if not ratios_w or not ratios_h:
        return {}
    return {'base_rw': float(np.median(ratios_w)), 'base_rh': float(np.median(ratios_h)), 'p90_rw': float(np.percentile(ratios_w, TRAINING_ROI_CAP_PERCENTILE)), 'p90_rh': float(np.percentile(ratios_h, TRAINING_ROI_CAP_PERCENTILE)), 'n_training_anchors': int(len(ratios_w))}


def canonical_box(cx: float, cy: float, crop_w: int, crop_h: int, w: int, h: int) -> Tuple[int, int, int, int]:
    crop_w = min(max(16, int(crop_w)), w)
    crop_h = min(max(16, int(crop_h)), h)
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(max(w - crop_w, 0), x1))
    y1 = max(0, min(max(h - crop_h, 0), y1))
    return (x1, y1, x1 + crop_w, y1 + crop_h)


def choose_video_crop_size(detections: List[Dict[str, float]], w: int, h: int, prior: Dict[str, float]) -> Tuple[int, int, str, float]:
    bw = np.asarray([d['box_w'] for d in detections if np.isfinite(d.get('box_w', np.nan))], float)
    bh = np.asarray([d['box_h'] for d in detections if np.isfinite(d.get('box_h', np.nan))], float)
    if prior:
        base_rw, base_rh = (prior['base_rw'], prior['base_rh'])
        base_w, base_h = (max(1.0, base_rw * w), max(1.0, base_rh * h))
        train_cap = min(prior['p90_rw'] / max(base_rw, 1e-08), prior['p90_rh'] / max(base_rh, 1e-08))
        allowed = max(1.0, min(float(MAX_ROI_SCALE_FROM_TRAIN_MEDIAN), float(train_cap)))
        requested = 1.0
        if len(bw) and len(bh):
            requested = max(1.0, float(np.percentile(bw, VIDEO_YOLO_BOX_PERCENTILE)) * YOLO_TO_CROP_SCALE / base_w, float(np.percentile(bh, VIDEO_YOLO_BOX_PERCENTILE)) * YOLO_TO_CROP_SCALE / base_h)
        applied = min(requested, allowed)
        crop_w = int(round(base_w * applied))
        crop_h = int(round(base_h * applied))
        source = 'training_median' if applied <= 1.0001 else 'training_median_mild_video_enlargement'
        return (min(w, max(32, crop_w)), min(h, max(32, crop_h)), source, float(applied))
    if not len(bw) or not len(bh):
        raise RuntimeError('No training ROI prior and no valid YOLO boxes.')
    crop_w = int(round(np.median(bw) * YOLO_TO_CROP_SCALE))
    crop_h = int(round(np.median(bh) * YOLO_TO_CROP_SCALE))
    return (min(w, max(32, crop_w)), min(h, max(32, crop_h)), 'yolo_fallback', 1.0)


def uniform_candidate_indices(frame_count: int, n: int) -> List[int]:
    if frame_count <= 0:
        return []
    context_margin = max((abs(int(x)) for x in CONTEXT_OFFSETS))
    frac_margin = int(round((frame_count - 1) * EDGE_EXCLUDE_FRACTION))
    lo = max(context_margin, frac_margin)
    hi = min(frame_count - 1 - context_margin, frame_count - 1 - frac_margin)
    if hi < lo:
        lo, hi = (context_margin, frame_count - 1 - context_margin)
    if hi < lo:
        return []
    k = min(max(1, n), hi - lo + 1)
    return sorted(set(np.round(np.linspace(lo, hi, k)).astype(int).tolist()))


def read_selected_frames(video_path: Path, indices: Sequence[int]) -> Tuple[Dict[int, np.ndarray], int, int, float, int]:
    wanted = set((int(x) for x in indices if int(x) >= 0))
    out: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 1e-06:
        fps = 25.0
    idx = 0
    max_idx = max(wanted) if wanted else -1
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = frame.copy()
            if len(out) == len(wanted):
                break
        idx += 1
    cap.release()
    return (out, w, h, fps, nframes)


def choose_detection(boxes: List[Dict], prev_center: Optional[Tuple[float, float]], w: int, h: int) -> Optional[Dict]:
    boxes = [b for b in boxes if float(b.get('score', 0.0)) >= MIN_YOLO_CONFIDENCE]
    if not boxes:
        return None
    if prev_center is None:
        return max(boxes, key=lambda b: float(b.get('score', 0.0)))
    diag = max(math.hypot(w, h), 1.0)
    px, py = prev_center

    def rank(b: Dict) -> float:
        cx = 0.5 * (float(b['x1']) + float(b['x2']))
        cy = 0.5 * (float(b['y1']) + float(b['y2']))
        dist = math.hypot(cx - px, cy - py) / diag
        return float(b.get('score', 0.0)) - 0.6 * dist
    return max(boxes, key=rank)


def _copy_box(box: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return None if box is None else dict(box)


def box_area(box: Dict[str, Any]) -> float:
    return float(max(0.0, float(box['x2']) - float(box['x1'])) * max(0.0, float(box['y2']) - float(box['y1'])))


def box_center(box: Dict[str, Any]) -> Tuple[float, float]:
    return (0.5 * (float(box['x1']) + float(box['x2'])), 0.5 * (float(box['y1']) + float(box['y2'])))


def box_wh(box: Dict[str, Any]) -> Tuple[float, float]:
    return (float(box['x2'] - box['x1']), float(box['y2'] - box['y1']))


def box_iou(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a['x1']), float(b['x1']))
    y1 = max(float(a['y1']), float(b['y1']))
    x2 = min(float(a['x2']), float(b['x2']))
    y2 = min(float(a['y2']), float(b['y2']))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 1e-08 else 0.0


def valid_yolo_candidates(boxes: List[Dict], w: int, h: int) -> List[Dict[str, Any]]:
    out = []
    for b0 in boxes:
        b = dict(b0)
        score = float(b.get('score', 0.0))
        if score < MIN_YOLO_CONFIDENCE:
            continue
        try:
            bw, bh = box_wh(b)
        except Exception:
            continue
        if bw < 16 or bh < 16:
            continue
        aspect = max(bw / max(bh, 1.0), bh / max(bw, 1.0))
        if not np.isfinite(aspect) or aspect > 5.0:
            continue
        cx, cy = box_center(b)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        out.append(b)
    return out


def box_motion_metrics(candidate: Dict[str, Any], previous: Optional[Dict[str, Any]], w: int, h: int) -> Dict[str, float]:
    if previous is None:
        return {'temporal_iou': np.nan, 'center_shift_box_diag': np.nan, 'center_shift_image_diag': np.nan, 'box_size_ratio': np.nan}
    cx, cy = box_center(candidate)
    px, py = box_center(previous)
    dist = math.hypot(cx - px, cy - py)
    pw, ph = box_wh(previous)
    prev_diag = max(math.hypot(max(pw, 1.0), max(ph, 1.0)), 1.0)
    image_diag = max(math.hypot(max(w, 1), max(h, 1)), 1.0)
    return {'temporal_iou': box_iou(candidate, previous), 'center_shift_box_diag': float(dist / prev_diag), 'center_shift_image_diag': float(dist / image_diag), 'box_size_ratio': float(max(box_area(candidate), 1.0) / max(box_area(previous), 1.0))}


def is_temporally_consistent_box(candidate: Dict[str, Any], previous: Optional[Dict[str, Any]], w: int, h: int) -> Tuple[bool, str, Dict[str, float]]:
    m = box_motion_metrics(candidate, previous, w, h)
    if previous is None:
        return (True, 'no_previous', m)
    size_ratio = float(m['box_size_ratio'])
    if size_ratio < MIN_BOX_SIZE_RATIO or size_ratio > MAX_BOX_SIZE_RATIO:
        return (False, 'unstable_size_change', m)
    sudden_jump = float(m['temporal_iou']) < MIN_TEMPORAL_IOU and float(m['center_shift_box_diag']) > MAX_CENTER_SHIFT_BOX_DIAG_RATIO and (float(m['center_shift_image_diag']) > MAX_CENTER_SHIFT_IMAGE_DIAG_RATIO)
    if sudden_jump:
        return (False, 'unstable_spatial_jump', m)
    return (True, 'stable', m)


def predict_track_center(prev_idx: int, prev_box: Dict[str, Any], prevprev_idx: Optional[int], prevprev_box: Optional[Dict[str, Any]], target_idx: int) -> Tuple[float, float]:
    pcx, pcy = box_center(prev_box)
    if prevprev_box is None or prevprev_idx is None or prev_idx == prevprev_idx:
        return (pcx, pcy)
    qx, qy = box_center(prevprev_box)
    dt = float(prev_idx - prevprev_idx)
    if abs(dt) < 1e-06:
        return (pcx, pcy)
    vx, vy = ((pcx - qx) / dt, (pcy - qy) / dt)
    step = float(target_idx - prev_idx)
    return (pcx + vx * step, pcy + vy * step)


def trajectory_error_box_diag(candidate: Dict[str, Any], prev_idx: int, prev_box: Dict[str, Any], prevprev_idx: Optional[int], prevprev_box: Optional[Dict[str, Any]], target_idx: int) -> float:
    pred_x, pred_y = predict_track_center(prev_idx, prev_box, prevprev_idx, prevprev_box, target_idx)
    cx, cy = box_center(candidate)
    pw, ph = box_wh(prev_box)
    prev_diag = max(math.hypot(max(pw, 1.0), max(ph, 1.0)), 1.0)
    return float(math.hypot(cx - pred_x, cy - pred_y) / prev_diag)


def smooth_roi_box(previous: Optional[Dict[str, Any]], current: Dict[str, Any], alpha: float=ROI_SMOOTH_ALPHA) -> Dict[str, Any]:
    if previous is None:
        return _copy_box(current)
    out = _copy_box(current)
    for k in ['x1', 'y1', 'x2', 'y2']:
        out[k] = int(round(alpha * float(current[k]) + (1.0 - alpha) * float(previous[k])))
    out['score'] = float(current.get('score', 0.0))
    return out


def select_tracked_box(boxes: List[Dict], prev_idx: int, prev_box: Dict[str, Any], prevprev_idx: Optional[int], prevprev_box: Optional[Dict[str, Any]], target_idx: int, w: int, h: int) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, float], Optional[Dict[str, Any]]]:
    candidates = valid_yolo_candidates(boxes, w, h)
    if not candidates:
        return (None, 'missing', {'temporal_iou': np.nan, 'center_shift_box_diag': np.nan, 'center_shift_image_diag': np.nan, 'box_size_ratio': np.nan, 'trajectory_error_box_diag': np.nan}, None)
    scored = []
    rejected = []
    for b in candidates:
        ok, reason, m = is_temporally_consistent_box(b, prev_box, w, h)
        traj = trajectory_error_box_diag(b, prev_idx, prev_box, prevprev_idx, prevprev_box, target_idx)
        m = dict(m)
        m['trajectory_error_box_diag'] = float(traj)
        if ok and np.isfinite(traj) and (traj > MAX_TRAJECTORY_ERROR_BOX_DIAG_RATIO):
            ok, reason = (False, 'unstable_trajectory_jump')
        if ok:
            size_term = abs(math.log(max(float(m['box_size_ratio']), 1e-06))) if np.isfinite(m['box_size_ratio']) else 0.0
            score = float(b.get('score', 0.0)) + TRACK_IOU_WEIGHT * float(0.0 if pd.isna(m['temporal_iou']) else m['temporal_iou']) - TRACK_CENTER_PENALTY * float(0.0 if pd.isna(m['center_shift_box_diag']) else m['center_shift_box_diag']) - TRACK_TRAJECTORY_PENALTY * float(0.0 if pd.isna(traj) else traj) - TRACK_SIZE_PENALTY * size_term
            scored.append((score, b, m))
        else:
            rejected.append((reason, b, m))
    if not scored:
        if rejected:
            reason, b, m = sorted(rejected, key=lambda x: float(x[1].get('score', 0.0)), reverse=True)[0]
            return (None, reason, m, b)
        return (None, 'rejected_unstable', {}, None)
    _, raw_best, metrics = sorted(scored, key=lambda x: x[0], reverse=True)[0]
    return (smooth_roi_box(prev_box, raw_best), 'tracked', metrics, raw_best)


def best_neighbor_support(center_box: Dict[str, Any], neighbor_boxes: List[Dict], w: int, h: int) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, float]]:
    best = None
    best_score = -1000000000.0
    best_metrics: Dict[str, float] = {}
    for b in valid_yolo_candidates(neighbor_boxes, w, h):
        ok, _, m = is_temporally_consistent_box(b, center_box, w, h)
        if not ok:
            continue
        s = float(b.get('score', 0.0)) + TRACK_IOU_WEIGHT * float(0.0 if pd.isna(m['temporal_iou']) else m['temporal_iou']) - TRACK_CENTER_PENALTY * float(0.0 if pd.isna(m['center_shift_box_diag']) else m['center_shift_box_diag'])
        if s > best_score:
            best = _copy_box(b)
            best_score = float(s)
            best_metrics = dict(m)
    return (best, best_score, best_metrics)


def select_confirmed_center_anchor(boxes_by_frame: Dict[int, List[Dict]], center_idx: int, w: int, h: int, preferred_center: Optional[Tuple[float, float]]=None, preferred_box_diag: Optional[float]=None) -> Tuple[Optional[Dict[str, Any]], int, float, Dict[int, Dict[str, Any]]]:
    centers = valid_yolo_candidates(boxes_by_frame.get(center_idx, []), w, h)
    if not centers:
        return (None, 0, np.nan, {})
    scored = []
    for c in centers:
        support_boxes: Dict[int, Dict[str, Any]] = {}
        support_score = 0.0
        for ni in [center_idx - 1, center_idx + 1]:
            nb, s, m = best_neighbor_support(c, boxes_by_frame.get(ni, []), w, h)
            if nb is not None:
                support_boxes[ni] = nb
                support_score += float(s)
        support = len(support_boxes)
        if support >= CENTER_CONFIRM_MIN_NEIGHBORS:
            coarse_penalty = 0.0
            if preferred_center is not None:
                cx, cy = box_center(c)
                denom = max(float(preferred_box_diag or math.hypot(*box_wh(c))), 1.0)
                coarse_penalty = math.hypot(cx - preferred_center[0], cy - preferred_center[1]) / denom
            score = float(c.get('score', 0.0)) + 0.25 * support + 0.1 * support_score - CENTER_COARSE_PRIOR_PENALTY * coarse_penalty
            scored.append((score, c, support_boxes))
    if not scored:
        return (None, 0, np.nan, {})
    score, box, support_boxes = sorted(scored, key=lambda x: x[0], reverse=True)[0]
    return (_copy_box(box), int(len(support_boxes)), float(score), support_boxes)


def propagated_fallback_box(prev_idx: int, prev_box: Dict[str, Any], prevprev_idx: Optional[int], prevprev_box: Optional[Dict[str, Any]], target_idx: int) -> Dict[str, Any]:
    cx, cy = predict_track_center(prev_idx, prev_box, prevprev_idx, prevprev_box, target_idx)
    bw, bh = box_wh(prev_box)
    out = _copy_box(prev_box)
    out['x1'] = float(cx - bw / 2.0)
    out['x2'] = float(cx + bw / 2.0)
    out['y1'] = float(cy - bh / 2.0)
    out['y2'] = float(cy + bh / 2.0)
    out['score'] = np.nan
    return out


def bidirectional_local_track(boxes_by_frame: Dict[int, List[Dict]], center_idx: int, anchor: Dict[str, Any], seed_neighbors: Dict[int, Dict[str, Any]], w: int, h: int) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {center_idx: {'box': _copy_box(anchor), 'raw_box': _copy_box(anchor), 'valid': 1, 'source': 'center_confirmed', 'metrics': {'temporal_iou': 1.0, 'center_shift_box_diag': 0.0, 'center_shift_image_diag': 0.0, 'box_size_ratio': 1.0, 'trajectory_error_box_diag': 0.0}}}
    for direction in (-1, 1):
        first_idx = center_idx + direction
        opposite_idx = center_idx - direction
        seed = seed_neighbors.get(first_idx)
        opposite_seed = seed_neighbors.get(opposite_idx)
        if seed is not None:
            m = box_motion_metrics(seed, anchor, w, h)
            m['trajectory_error_box_diag'] = 0.0
            stable_seed = smooth_roi_box(anchor, seed)
            result[first_idx] = {'box': stable_seed, 'raw_box': _copy_box(seed), 'valid': 1, 'source': 'center_support_seed', 'metrics': m}
            prevprev_idx, prevprev_box = (center_idx, _copy_box(anchor))
            prev_idx, prev_box = (first_idx, stable_seed)
            start_step = 2
            missed = 0
        else:
            prev_idx, prev_box = (center_idx, _copy_box(anchor))
            if opposite_seed is not None:
                prevprev_idx, prevprev_box = (opposite_idx, _copy_box(opposite_seed))
            else:
                prevprev_idx, prevprev_box = (None, None)
            start_step = 1
            missed = 0
        for step in range(start_step, TRACK_RADIUS_FRAMES + 1):
            fi = center_idx + direction * step
            stable, source, metrics, raw = select_tracked_box(boxes_by_frame.get(fi, []), prev_idx, prev_box, prevprev_idx, prevprev_box, fi, w, h)
            if stable is not None:
                result[fi] = {'box': stable, 'raw_box': _copy_box(raw), 'valid': 1, 'source': source, 'metrics': metrics}
                prevprev_idx, prevprev_box = (prev_idx, prev_box)
                prev_idx, prev_box = (fi, stable)
                missed = 0
            else:
                missed += 1
                fallback = propagated_fallback_box(prev_idx, prev_box, prevprev_idx, prevprev_box, fi)
                result[fi] = {'box': fallback, 'raw_box': _copy_box(raw), 'valid': 0, 'source': source if source else 'missing_or_rejected', 'metrics': metrics}
                if missed > MAX_TRACK_GAP_FRAMES:
                    for rest in range(step + 1, TRACK_RADIUS_FRAMES + 1):
                        fj = center_idx + direction * rest
                        fb = propagated_fallback_box(prev_idx, prev_box, prevprev_idx, prevprev_box, fj)
                        result[fj] = {'box': fb, 'raw_box': None, 'valid': 0, 'source': 'track_terminated_after_gap', 'metrics': {'temporal_iou': np.nan, 'center_shift_box_diag': np.nan, 'center_shift_image_diag': np.nan, 'box_size_ratio': np.nan, 'trajectory_error_box_diag': np.nan}}
                    break
    return result


def add_local_geometry_change(records: List[Dict[str, float]]) -> None:
    if not records:
        return
    order = sorted(range(len(records)), key=lambda i: records[i]['frame_idx'])
    log_area = np.asarray([math.log(max(records[i]['box_w'] * records[i]['box_h'], 1.0)) for i in order], dtype=np.float32)
    log_aspect = np.asarray([math.log(max(records[i]['box_w'] / max(records[i]['box_h'], 1.0), 1e-06)) for i in order], dtype=np.float32)
    for pos, ridx in enumerate(order):
        vals = []
        if pos > 0:
            vals.append(abs(float(log_area[pos] - log_area[pos - 1])) + 0.5 * abs(float(log_aspect[pos] - log_aspect[pos - 1])))
        if pos + 1 < len(order):
            vals.append(abs(float(log_area[pos + 1] - log_area[pos])) + 0.5 * abs(float(log_aspect[pos + 1] - log_aspect[pos])))
        records[ridx]['geometry_change'] = float(max(vals) if vals else 0.0)


def greedy_diverse_indices(records: List[Dict[str, float]], k: int) -> List[int]:
    if len(records) <= k:
        return list(range(len(records)))
    vals = []
    for r in records:
        area = max(r['box_w'] * r['box_h'], 1.0)
        aspect = r['box_w'] / max(r['box_h'], 1.0)
        vals.append([r['time_fraction'], math.log(area + 1.0), math.log(aspect + 1e-06), r['center_x_norm'], r['center_y_norm'], math.log(r['blur_score'] + 1.0), float(r.get('geometry_change', 0.0))])
    X = np.asarray(vals, np.float32)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    X = (X - med) / np.maximum(1.4826 * mad, 0.25)
    first = int(np.argmax([r['yolo_confidence'] for r in records]))
    selected = [first]
    min_dist = np.linalg.norm(X - X[first], axis=1)
    while len(selected) < k:
        min_dist[selected] = -1
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] < 0:
            break
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(X - X[nxt], axis=1))
    return sorted(selected, key=lambda i: records[i]['frame_idx'])
