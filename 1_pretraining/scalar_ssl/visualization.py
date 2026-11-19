"""Representation probes, context attention and training figures (methodology preserved)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import settings
from .settings import *
from . import cache as _ssl_mod_cache
from . import losses as _ssl_mod_losses
from . import model as _ssl_mod_model
from . import utils as _ssl_mod_utils


def select_probe_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if 'center_frame_idx' in df.columns:
        frame_col = 'center_frame_idx'
    elif 'frame_idx' in df.columns:
        frame_col = 'frame_idx'
    else:
        raise KeyError("Probe manifest must contain 'center_frame_idx' (schema v3) or legacy 'frame_idx'.")
    one = df.sort_values(['video_uid', frame_col]).groupby('video_uid', as_index=False).first()
    rng = np.random.default_rng(seed)
    if len(one) <= n:
        return one.reset_index(drop=True)
    idx = rng.choice(len(one), n, replace=False)
    return one.iloc[sorted(idx)].reset_index(drop=True)


@torch.no_grad()
def extract_backbone_features(backbone: nn.Module, paths: Sequence[str], device: torch.device, batch_size: int=6) -> Tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    embeddings, maps = ([], [])
    for start in range(0, len(paths), batch_size):
        xs = []
        for p in paths[start:start + batch_size]:
            bgr = _ssl_mod_utils.read_image(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xs.append(_ssl_mod_utils.image_tensor(rgb))
        x = torch.stack(xs).to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP and device.type == 'cuda'):
            feat = backbone.get_intermediate_layers(x, n=[11], reshape=True)[0]
        emb = feat.mean(dim=(2, 3))
        mag = torch.linalg.vector_norm(feat.float(), dim=1)
        embeddings.append(emb.float().cpu().numpy())
        maps.append(mag.cpu().numpy())
    return (np.concatenate(embeddings, axis=0), np.concatenate(maps, axis=0))


def effective_rank(emb: np.ndarray) -> float:
    X = emb.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    if len(X) < 2:
        return np.nan
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    p = s ** 2 / max(np.sum(s ** 2), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-08)
    return np.sum(an * bn, axis=1)


def feature_heatmap(m: np.ndarray, size: int=IMAGE_SIZE) -> np.ndarray:
    x = m.astype(np.float32)
    lo, hi = np.percentile(x, [5, 95])
    x = np.clip((x - lo) / max(hi - lo, 1e-06), 0, 1)
    x = cv2.resize((x * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_feature_map_panel(probe_df: pd.DataFrame, before_maps: np.ndarray, after_maps: np.ndarray, out_path: Path, max_cases: int=8) -> None:
    rows = []
    for i in range(min(max_cases, len(probe_df))):
        bgr = _ssl_mod_utils.read_image(probe_df.iloc[i]['cache_path'])
        bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE))
        hm0 = feature_heatmap(before_maps[i])
        hm1 = feature_heatmap(after_maps[i])
        diff = np.abs(after_maps[i].astype(np.float32) - before_maps[i].astype(np.float32))
        hmd = feature_heatmap(diff)
        row = np.concatenate([put_label(bgr, 'ROI'), put_label(hm0, 'DINO before: feature magnitude'), put_label(hm1, 'After SSL: feature magnitude'), put_label(hmd, 'Absolute feature change')], axis=1)
        rows.append(row)
    if rows:
        _ssl_mod_utils.write_image(out_path, np.concatenate(rows, axis=0), quality=94)


def nearest_neighbor_indices(emb: np.ndarray, qidx: int, topk: int) -> List[int]:
    x = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-08)
    sim = x @ x[qidx]
    sim[qidx] = -np.inf
    return np.argsort(-sim)[:topk].tolist()


def save_neighbor_panel(gallery_df: pd.DataFrame, before_emb: np.ndarray, after_emb: np.ndarray, out_path: Path, queries: int=NN_QUERY_CASES, topk: int=NN_TOPK) -> None:
    if len(gallery_df) < topk + 2:
        return
    q_indices = np.linspace(0, len(gallery_df) - 1, min(queries, len(gallery_df)), dtype=int).tolist()
    rows = []
    thumb = VIS_THUMB_SIZE
    for q in q_indices:
        base_nn = nearest_neighbor_indices(before_emb, q, topk)
        adapt_nn = nearest_neighbor_indices(after_emb, q, topk)
        inds = [q] + base_nn + adapt_nn
        labels = ['QUERY'] + [f'before NN{i + 1}' for i in range(topk)] + [f'after NN{i + 1}' for i in range(topk)]
        tiles = []
        for idx, lab in zip(inds, labels):
            img = _ssl_mod_utils.read_image(gallery_df.iloc[idx]['cache_path'])
            img = cv2.resize(img, (thumb, thumb))
            tiles.append(put_label(img, lab))
        rows.append(np.concatenate(tiles, axis=1))
    _ssl_mod_utils.write_image(out_path, np.concatenate(rows, axis=0), quality=94)


def save_training_plot(history: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'train_total_loss' in history:
            ax.plot(history['epoch'], history['train_total_loss'], marker='o', label='train total')
        if 'val_total_loss' in history:
            ax.plot(history['epoch'], history['val_total_loss'], marker='o', label='monitor total')
        if 'train_frame_loss' in history:
            ax.plot(history['epoch'], history['train_frame_loss'], alpha=0.75, label='train frame SSL')
        if 'train_context_loss' in history:
            ax.plot(history['epoch'], history['train_context_loss'], alpha=0.75, label='train context')
        if 'val_context_loss' in history:
            ax.plot(history['epoch'], history['val_context_loss'], alpha=0.65, label='monitor context')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Self-supervised loss')
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f'Visualization warning: could not save training plot: {e}')


def raw_clip_tensor_from_row(row: pd.Series, device: torch.device) -> torch.Tensor:
    xs = []
    for c in _ssl_mod_cache.clip_path_columns():
        bgr = _ssl_mod_utils.read_image(str(row[c]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xs.append(_ssl_mod_utils.image_tensor(rgb))
    return torch.stack(xs, dim=0).to(device)


@torch.no_grad()
def save_context_attention_artifacts(student: _ssl_mod_model.SSLStudent, teacher: _ssl_mod_model.SSLTeacher, context_head: _ssl_mod_model.CrossSectionContextHead, manifest: pd.DataFrame, device: torch.device) -> None:
    if manifest.empty:
        return
    vis_dir = settings.OUTPUT_ROOT / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    cases = select_probe_df(manifest, CONTEXT_VIS_CASES, SEED + 909)
    rows_csv = []
    panel_rows = []
    ctx_offsets = [o for o in CONTEXT_OFFSETS if o != 0]
    student.eval()
    teacher.eval()
    context_head.eval()
    for _, row in cases.iterrows():
        x = raw_clip_tensor_from_row(row, device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP and device.type == 'cuda'):
            sproj, _ = student(x)
            tproj = teacher(x)
            tracking_valid = np.asarray([float(row.get(f'frame_{off:+d}_tracking_valid', 1.0 if off == 0 else 0.0)) for off in CONTEXT_OFFSETS], dtype=np.float32)
            ctx_valid = np.delete(tracking_valid, CENTER_INDEX)
            mask = torch.from_numpy(ctx_valid[None]).to(device)
            enabled = torch.tensor([float(ctx_valid.sum() >= MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS)], device=device, dtype=torch.float32)
            _, alpha = _ssl_mod_losses.context_prediction_loss(sproj, tproj, context_head, mask, enabled, 1, len(CONTEXT_OFFSETS))
            sg = sproj[-1].mean(dim=(2, 3)).view(1, len(CONTEXT_OFFSETS), -1)
            tg = tproj[-1].mean(dim=(2, 3)).view(1, len(CONTEXT_OFFSETS), -1)
            ctx_idx = [i for i in range(len(CONTEXT_OFFSETS)) if i != CENTER_INDEX]
            pred_center, alpha_last = context_head(sg[:, ctx_idx], mask)
            cosine = F.cosine_similarity(pred_center, tg[:, CENTER_INDEX], dim=-1).item() if float(enabled.item()) > 0.5 else np.nan
        a = alpha[0].float().cpu().numpy()
        rec = {'clip_uid': row['clip_uid'], 'video_uid': row['video_uid'], 'center_frame_idx': int(row['center_frame_idx']), 'pred_center_cosine': float(cosine)}
        for off, val in zip(ctx_offsets, a):
            rec[f'alpha_{off:+d}'] = float(val)
            rec[f'tracking_valid_{off:+d}'] = int(row.get(f'frame_{off:+d}_tracking_valid', 0))
        rows_csv.append(rec)
        tiles = []
        ctx_map = {off: float(val) for off, val in zip(ctx_offsets, a)}
        for off, c in zip(CONTEXT_OFFSETS, _ssl_mod_cache.clip_path_columns()):
            img = _ssl_mod_utils.read_image(str(row[c]))
            img = cv2.resize(img, (VIS_THUMB_SIZE, VIS_THUMB_SIZE))
            if off == 0:
                cos_text = f'{cosine:.2f}' if np.isfinite(cosine) else 'NA'
                lab = f't center | confirmed | cos={cos_text}'
            else:
                tv = int(row.get(f'frame_{off:+d}_tracking_valid', 0))
                lab = f't{off:+d} | valid={tv} | alpha={ctx_map[off]:.2f}'
            tiles.append(put_label(img, lab))
        panel_rows.append(np.concatenate(tiles, axis=1))
    if rows_csv:
        pd.DataFrame(rows_csv).to_csv(vis_dir / 'context_attention_examples.csv', index=False, encoding='utf-8-sig')
    if panel_rows:
        _ssl_mod_utils.write_image(vis_dir / 'context_attention_examples.jpg', np.concatenate(panel_rows, axis=0), quality=94)


def build_probe_artifacts_before_training(backbone: nn.Module, manifest: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    vis_dir = settings.OUTPUT_ROOT / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    gallery = select_probe_df(manifest, NN_GALLERY_CASES, SEED + 501)
    probe = gallery.iloc[:min(PROBE_CASES, len(gallery))].copy().reset_index(drop=True)
    gallery.to_csv(vis_dir / 'probe_gallery.csv', index=False, encoding='utf-8-sig')
    probe.to_csv(vis_dir / 'probe_cases.csv', index=False, encoding='utf-8-sig')
    before_gallery_emb, before_gallery_maps = extract_backbone_features(backbone, gallery['cache_path'].astype(str).tolist(), device)
    np.savez_compressed(vis_dir / 'baseline_probe_features.npz', gallery_embeddings=before_gallery_emb, gallery_maps=before_gallery_maps)
    return {'gallery': gallery, 'probe': probe, 'before_gallery_emb': before_gallery_emb, 'before_gallery_maps': before_gallery_maps}


def finalize_probe_artifacts(backbone: nn.Module, probe_info: Dict[str, Any], device: torch.device) -> None:
    vis_dir = settings.OUTPUT_ROOT / 'visualization'
    gallery = probe_info['gallery']
    before_emb = probe_info['before_gallery_emb']
    before_maps = probe_info['before_gallery_maps']
    after_emb, after_maps = extract_backbone_features(backbone, gallery['cache_path'].astype(str).tolist(), device)
    np.savez_compressed(vis_dir / 'adapted_probe_features.npz', gallery_embeddings=after_emb, gallery_maps=after_maps)
    n_probe = min(PROBE_CASES, len(gallery))
    drift = cosine_rows(before_emb, after_emb)
    metrics = pd.DataFrame([{'n_gallery': len(gallery), 'mean_before_after_cosine': float(np.mean(drift)), 'std_before_after_cosine': float(np.std(drift)), 'effective_rank_before': effective_rank(before_emb), 'effective_rank_after': effective_rank(after_emb), 'mean_feature_norm_before': float(np.mean(np.linalg.norm(before_emb, axis=1))), 'mean_feature_norm_after': float(np.mean(np.linalg.norm(after_emb, axis=1)))}])
    metrics.to_csv(vis_dir / 'probe_representation_metrics.csv', index=False, encoding='utf-8-sig')
    save_feature_map_panel(gallery.iloc[:n_probe].reset_index(drop=True), before_maps[:n_probe], after_maps[:n_probe], vis_dir / 'feature_maps_before_after.jpg')
    save_neighbor_panel(gallery.reset_index(drop=True), before_emb, after_emb, vis_dir / 'nearest_neighbors_before_after.jpg')
    with open(vis_dir / 'README_visualization.txt', 'w', encoding='utf-8') as f:
        f.write('1) feature_maps_before_after.jpg: patch-feature magnitude, NOT a saliency/attention map.\n   Use it only to see where the backbone representation changed spatially.\n2) nearest_neighbors_before_after.jpg: qualitative retrieval from a fixed unlabeled gallery.\n   Useful for checking whether similar carotid appearances become more coherent after SSL.\n3) probe_representation_metrics.csv: representation drift and effective-rank checks.\n   A cosine near 1 means little adaptation; a very low cosine would suggest excessive drift.\n4) ssl_training_curves.png: teacher-student consistency loss only; downstream segmentation\n   performance must still be judged by the existing supervised image/V5 validation.\n')
