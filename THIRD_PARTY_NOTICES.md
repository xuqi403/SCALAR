# Third-party code, models and data

SCALAR's root `LICENSE` is intended only for original code that its copyright owners are authorized to license. It **does not grant rights** to third-party software, underlying model architectures/weights, ultrasound datasets, annotations, or clinical records.

## DINOv3

- Upstream: https://github.com/facebookresearch/dinov3
- Governing license: [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), **not Apache-2.0**.
- The upstream source and original model weights are not bundled in this repository. Obtain and use them under Meta's terms, and acknowledge them as required.
- Any carotid-adapted DINOv3 checkpoint may be subject to the upstream license and institutional/weight-ownership permissions; the root Apache-2.0 license does not independently authorize redistribution of such weights.

## Carotid ROI detector / YOLO

- SCALAR expects the compatible ROI detector interface described in the README (e.g., `yolo_roi.py` and `utils/utils.py`). Neither the original detector code nor weights are included here.
- **Verify which implementation and license was actually used** before adding its code or checkpoint. Different YOLO implementations use different licenses.
- In particular, current Ultralytics YOLO releases have AGPL-3.0 / commercial licensing options (https://www.ultralytics.com/license), but this does **not** prove that the detector used by this study was Ultralytics.

## Third-party Python dependencies

PyTorch, torchvision, OpenCV, NumPy, SciPy, pandas, Pillow, Matplotlib, tqdm and pytest remain subject to their respective licenses. Installing them as dependencies does not sublicense them under SCALAR's license.

## Pre-publication checklist

1. Confirm ownership/authorization to license the original SCALAR source (including institutional and co-author approvals).
2. Audit imported or copied code for third-party copyright notices and obligations.
3. Verify exact DINOv3 and detector versions/terms and any proposed checkpoint redistribution.
4. Exclude private videos, images, patient identifiers, real manifests, cache files, checkpoints and any such material in **Git history**, not only the current working tree.
5. Record the exact manuscript checkpoints and keep internal data/split audit records.
