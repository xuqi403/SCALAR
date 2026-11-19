from pathlib import Path
import csv
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort import load_approved_manifest


def write_manifest(p, rows):
    with p.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["video_path", "patient_id"])
        w.writeheader()
        w.writerows(rows)


def test_approved_subset(tmp_path):
    a=tmp_path / "a.mp4"; b=tmp_path / "b.mp4"
    a.write_bytes(b"fake"); b.write_bytes(b"fake")
    manifest=tmp_path / "internal.csv"
    write_manifest(manifest,[{"video_path":"a.mp4", "patient_id":"A"},
                             {"video_path":"b.mp4", "patient_id":"B"}])
    rows=load_approved_manifest(manifest,expected_videos=2)
    assert len(rows)==2 and all(Path(x["video_path"]).is_absolute() for x in rows)


def test_wrong_cohort_size_fails(tmp_path):
    p=tmp_path / "internal.csv"
    write_manifest(p,[{"video_path":"a.mp4", "patient_id":"A"}])
    with pytest.raises(ValueError,match="Expected exactly"):
        load_approved_manifest(p, expected_videos=5288)


def test_duplicate_path_fails(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"fake")
    p=tmp_path / "internal.csv"
    write_manifest(p,[{"video_path":"a.mp4", "patient_id":"A"},
                      {"video_path":"./a.mp4", "patient_id":"B"}])
    with pytest.raises(ValueError,match="Duplicate"):
        load_approved_manifest(p, expected_videos=2)


def test_excluded_patient_fails(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"fake")
    (tmp_path / "b.mp4").write_bytes(b"fake")
    p=tmp_path / "internal.csv"
    write_manifest(p,[{"video_path":"a.mp4", "patient_id":"A"},
                      {"video_path":"b.mp4", "patient_id":"B"}])
    exc=tmp_path / "excluded.csv"
    write_manifest(exc,[{"video_path":"unavailable.mp4", "patient_id":"B"}])
    with pytest.raises(ValueError,match="overlaps excluded"):
        load_approved_manifest(p,expected_videos=2,excluded_manifests=[exc])
