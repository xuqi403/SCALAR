import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalar_infer import io_utils, roi
import infer_video


class SegmentationOnlyTests(unittest.TestCase):
    def test_native_mask_restore(self):
        mask = np.zeros((448, 448), np.uint8)
        mask[110:330, 110:330] = 1
        mask[160:280, 160:280] = 2
        restored = roi.mask_448_to_native(mask, (10, 20, 458, 468), (520, 520))
        self.assertEqual(restored.shape, (520, 520))
        self.assertEqual(int(restored[220, 220]), 2)
        self.assertEqual(int(restored[0, 0]), 0)

    def test_gap_interpolation(self):
        arr = np.array([10.0, np.nan, 12.0])
        filled, mask = roi._fill_short_gaps(arr, np.isfinite(arr), 1)
        self.assertAlmostEqual(filled[1], 11.0)
        self.assertTrue(mask[1])

    def test_no_physical_cli(self):
        args = infer_video.parse_args([
            "--video", "fake.mp4", "--checkpoint", "fake.pth",
            "--dino-repo", "repo", "--dino-weights", "weights.pth", "--yolo-code-dir", "yolo"
        ])
        self.assertFalse(any(x in vars(args) for x in ("mm_per_pixel", "ruler_pixels", "scale_manifest")))

    def test_compact_has_no_area_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({
                "frame_idx": [0], "time_sec": [0.0], "track_valid": [1],
                "inference_valid": [1], "roi_x1": [0], "roi_y1": [0],
                "roi_x2": [448], "roi_y2": [448],
                "prediction_confidence": [0.95], "track_status": ["detected"],
            })
            out = io_utils.compact_frame_table(df, Path(tmp)/"video.mp4", Path(tmp), 448, 448, 25.0)
            self.assertFalse(any("area" in c or "thickness" in c or "mm" in c for c in out.columns))


if __name__ == "__main__":
    unittest.main()
