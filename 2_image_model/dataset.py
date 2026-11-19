"""De-identified image-model data interface.

The public training code consumes a caller-prepared CSV with explicit
train/val/test values. It neither searches clinical records nor assigns
patients to cohorts. Rows/paths/IDs are never copied into published outputs.

Labelme `wall` is the OUTER enclosed boundary (outer = wall region + lumen).
Labelme `lumen` is the LUMEN enclosed boundary.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
IMAGENET_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD = np.asarray(IMAGENET_STD, dtype=np.float32)

def read_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return image



def polygon_mask(points, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    if points is None or len(points) < 3:
        return out
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 1)
    return out



def load_labelme_mask(json_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    outer = np.zeros((h, w), dtype=np.uint8)
    lumen = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        poly = polygon_mask(pts, h, w)
        if label == "wall":
            outer = np.maximum(outer, poly)
        elif label == "lumen":
            lumen = np.maximum(lumen, poly)
    if outer.sum() == 0 or lumen.sum() == 0:
        raise ValueError(f"Missing wall/lumen polygon: {json_path}")
    lumen &= outer
    target = np.zeros((h, w), dtype=np.uint8)
    target[outer > 0] = 1
    target[lumen > 0] = 2
    return target



def letterbox(image: np.ndarray, mask: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = min(out_size / w, out_size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    left = (out_size - nw) // 2
    right = out_size - nw - left
    top = (out_size - nh) // 2
    bottom = out_size - nh - top
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image, mask



def image_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()



def ultrasound_augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.7:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-8, 8), random.uniform(0.95, 1.05))
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = image.astype(np.float32) / 255.0
    if random.random() < 0.5:
        x = np.power(np.clip(x, 0, 1), random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        x = x * random.uniform(0.88, 1.12) + random.uniform(-0.04, 0.04)
    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(0.015, 0.045), size=x.shape[:2]).astype(np.float32)
        x = x * (1 + noise[..., None])
    image = np.clip(x * 255, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=random.uniform(0.35, 0.8))
    return image, mask



def read_manifest(csv_file: str | Path, check_group_split: bool = True) -> pd.DataFrame:
    """Read an externally curated split manifest; NEVER create or change splits.

    Required: split, annotation_image_path, annotation_json_path.
    Optional: group_id for in-memory cross-split leakage screening, sample_id.
    Paths may be absolute, or relative to the CSV file's directory.
    """
    path = Path(csv_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ("split", "annotation_image_path", "annotation_json_path")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest lacks columns: {missing}")
    df = df.copy()
    df["split"] = df["split"].str.strip().str.lower()
    if df.empty or not df["split"].isin(("train", "val", "test")).all():
        raise ValueError("Manifest must be non-empty with only train, val, test split values")
    if not (df["split"].eq("train").any() and df["split"].eq("val").any()):
        raise ValueError("At least one train and one val frame required")
    if check_group_split and "group_id" in df.columns:
        known = df[df["group_id"].astype(str).str.strip().ne("")]
        overlaps = known.groupby("group_id")["split"].nunique()
        if (overlaps > 1).any():
            raise ValueError("group_id appears in multiple splits; fix this in the private data manifest")
    # This detects duplicate image/annotation pairs placed into different splits.
    keys = df["annotation_image_path"].str.replace("\\", "/", regex=False).str.casefold()
    dup = df.groupby(keys)["split"].nunique()
    if (dup > 1).any():
        raise ValueError("The same image appears in multiple splits")
    for col in ("annotation_image_path", "annotation_json_path"):
        resolved = []
        for value in df[col].tolist():
            if not str(value).strip():
                raise ValueError(f"Empty path in column {col}")
            p = Path(value).expanduser()
            if not p.is_absolute():
                p = path.parent / p
            if not p.is_file():
                raise FileNotFoundError(f"Missing {col}: {p}")
            resolved.append(str(p.resolve()))
        df[col] = resolved
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"image_{i:06d}" for i in range(len(df))]
    return df


class CarotidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool, augmentation: str, image_size: int = IMAGE_SIZE):
        self.df = df.reset_index(drop=True).copy()
        self.training = training
        self.augmentation = augmentation
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row["annotation_image_path"])
        json_path = str(row["annotation_json_path"])
        bgr = read_image(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = load_labelme_mask(json_path, rgb.shape[:2])
        if self.training:
            if self.augmentation == "ultrasound":
                rgb, mask = ultrasound_augment(rgb, mask)
        rgb, mask = letterbox(rgb, mask, self.image_size)
        return {"image": image_tensor(rgb),
                "mask": torch.from_numpy(mask.astype(np.int64))}
