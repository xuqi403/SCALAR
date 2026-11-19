"""Create non-patient illustrative shapes; NOT a scientific benchmark.
Run from the 2_image_model folder:
    python examples/make_synthetic_data.py --out-dir /tmp/scalar_image_demo
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import cv2
import numpy as np


def ellipse_points(cx, cy, ax, ay, n=36):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [[float(cx+ax*np.cos(t)), float(cy+ay*np.sin(t))] for t in theta]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, split in enumerate(("train", "train", "train", "val", "test")):
        img = np.zeros((112, 112, 3), np.uint8)
        img[:] = 45+idx*4
        cx, cy = 56+(idx%2)*2, 56
        cv2.ellipse(img, (cx, cy), (30, 33), 0, 0, 360, (140,140,140), -1)
        cv2.ellipse(img, (cx, cy), (22, 25), 0, 0, 360, (32,32,32), -1)
        path = out/f"synthetic_{idx}.png"
        cv2.imwrite(str(path), img)
        ann = {
            "imagePath": path.name,
            "shapes": [
                {"label":"wall","shape_type":"polygon","points":ellipse_points(cx,cy,30,33)},
                {"label":"lumen","shape_type":"polygon","points":ellipse_points(cx,cy,22,25)}
            ]
        }
        jp = out/f"synthetic_{idx}.json"
        jp.write_text(json.dumps(ann), encoding="utf-8")
        records.append({"split":split,"annotation_image_path":path.name,
                        "annotation_json_path":jp.name,"group_id":f"synthetic_{idx}",
                        "sample_id":f"synthetic_{idx}"})
    with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader();writer.writerows(records)
    print("Created synthetic data:",out/"manifest.csv")

if __name__ == "__main__":
    main()
