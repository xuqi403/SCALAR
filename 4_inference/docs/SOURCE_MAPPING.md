# Modular source mapping

The original release was organized as `infer_video.py` + a large `core.py`.
For readability, the **segmentation pipeline is retained** but grouped by responsibility:

| New module | Origin and scope |
|---|---|
| `infer_video.py` | CLI argument parsing, single/batch video selection, exit status |
| `scalar_infer/model_loader.py` | Original training module integration, V5 checkpoint validation |
| `scalar_infer/roi.py` | YOLO detection, ROI trajectory, fixed crop geometry, mask restoration |
| `scalar_infer/feature_bank.py` | Batched DINOv3 features with temporary float16 feature cache |
| `scalar_infer/segmentation.py` | Five-frame availability, V5 fusion, decoder and topology cleanup |
| `scalar_infer/overlay.py` | Optional native resolution contour-only video |
| `scalar_infer/io_utils.py` | Path handling, output metadata, CSV schema |
| `scalar_infer/pipeline.py` | Single-video orchestration, resume/checkpoint checks, artifact writes |
| `scalar_infer/settings.py` | Original inference constants |

No pixel-to-mm calibration, physical measurements, wall thickness, area profiles,
unwrapped maps or manuscript Figure 5 code is included. The mask stays
in 448×448 ROI coordinates and can be mapped back using `roi.mask_448_to_native`.

Model computations are transferred from the prior `core.py` without intentional
numerical modifications. Two batch-size settings are passed as explicit function
arguments rather than changed as global mutable variables.
