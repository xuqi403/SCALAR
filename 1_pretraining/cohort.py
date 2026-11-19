"""Validate a PRIVATE, explicit internal-only SSL video manifest.

This does NOT infer centre or patient provenance: membership must be curated and
verified against the institution's original split/hold-out records.
No clinical data or patient identifiers should ever be committed to GitHub.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict, Optional, Sequence

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "video_path" not in reader.fieldnames:
            raise ValueError(f"Manifest must contain a video_path column: {path}")
        return list(reader)


def _canonical(raw_path: str, video_root: Path) -> Path:
    raw = (raw_path or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Empty video_path in cohort manifest")
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else video_root / p).resolve()


def load_approved_manifest(
    path: Path,
    video_root: Optional[Path] = None,
    expected_videos: int = 5288,
    excluded_manifests: Sequence[Path] = (),
    check_files: bool = True,
) -> List[Dict[str, str]]:
    """Fail on count mismatch, missing paths, duplicate videos or excluded cohorts.

    `excluded_manifests` is optional and supports exclusions by video_path and,
    when both files contain the field, patient_id. It is not a substitute for
    institution-verified internal-centre cohort membership.
    """
    path = Path(path).expanduser().resolve()
    root = Path(video_root).expanduser().resolve() if video_root else path.parent
    rows = _read_csv(path)
    if len(rows) != expected_videos:
        raise ValueError(f"Expected exactly {expected_videos} listed videos, found {len(rows)} in {path}")
    approved: List[Dict[str, str]] = []
    seen = set()
    for i, row in enumerate(rows, start=2):
        p = _canonical(row.get("video_path", ""), root)
        if p.suffix.lower() not in ALLOWED_EXT:
            raise ValueError(f"Unsupported video extension on row {i}: {p.suffix}")
        key = str(p).casefold()
        if key in seen:
            raise ValueError(f"Duplicate video path on row {i}: {p.name}")
        seen.add(key)
        if check_files and not p.is_file():
            raise FileNotFoundError(f"Video not found on row {i}: {p}")
        approved.append({"video_path": str(p), "patient_id": (row.get("patient_id") or "").strip()})
    approved_ids = {r["patient_id"] for r in approved if r["patient_id"]}
    for exc in excluded_manifests:
        exc = Path(exc).expanduser().resolve()
        excluded_paths = set()
        excluded_ids = set()
        for row in _read_csv(exc):
            if (row.get("video_path") or "").strip():
                excluded_paths.add(str(_canonical(row["video_path"], root)).casefold())
            if (row.get("patient_id") or "").strip():
                excluded_ids.add(row["patient_id"].strip())
        overlap = seen & excluded_paths
        patient_overlap = approved_ids & excluded_ids
        if overlap or patient_overlap:
            raise ValueError(f"Cohort overlaps excluded manifest {exc.name}: "
                             f"{len(overlap)} videos, {len(patient_overlap)} patient IDs")
    return approved


def write_private_resolved_manifest(rows: List[Dict[str, str]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_path", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path
