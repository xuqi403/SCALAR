#!/usr/bin/env python
"""Validate approved internal SSL cohort before launching pretraining."""
import argparse
from pathlib import Path
from cohort import load_approved_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort_manifest", type=Path, required=True)
    p.add_argument("--video_root", type=Path, default=None)
    p.add_argument("--excluded_manifest", type=Path, action="append", default=[])
    p.add_argument("--expected_videos", type=int, default=5288)
    args = p.parse_args()
    rows = load_approved_manifest(args.cohort_manifest, args.video_root,
                                 args.expected_videos, args.excluded_manifest)
    print(f"PASS: {len(rows):,} distinct existing video files in approved internal-only manifest.")
    print("NOTE: centre provenance must be verified against the private original records.")


if __name__ == "__main__":
    main()
