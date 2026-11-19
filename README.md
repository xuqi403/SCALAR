# SCALAR

**Adaptive cross-sectional context for carotid artery segmentation in routine ultrasound sweeps**

SCALAR is a research framework for segmenting carotid artery anatomy from routine transverse B-mode ultrasound sweeps. It uses a domain-adapted DINOv3 encoder and information from neighboring cross-sections to improve segmentation of the target frame, particularly when local boundaries are poorly visualized.

The repository contains the core code for self-supervised encoder adaptation, supervised segmentation training, adaptive cross-sectional fusion, and whole-video inference.

> **Research use only.** SCALAR is not a certified medical device and must not be used for clinical diagnosis or patient management.

## Method

SCALAR takes five neighboring cross-sections at offsets **−4, −2, 0, +2, +4** relative to the target frame. After vessel localization and ROI alignment, DINOv3 extracts frame-level features. An adaptive fusion module uses learned alignment, context reliability, and a spatially varying gate to combine target-frame and neighboring-frame information. A segmentation decoder predicts the outer vessel and lumen boundaries.

```text
Transverse ultrasound sweep
          │
          ▼
YOLO vessel localization and ROI tracking
          │
          ▼
Five-frame context (−4, −2, 0, +2, +4)
          │
          ▼
Carotid-adapted DINOv3 encoder
          │
          ▼
Adaptive feature alignment and context fusion
          │
          ▼
Outer-vessel and lumen segmentation
```

Inputs are resized to **448 × 448**. Segmentation labels are **0 = background**, **1 = vessel wall**, and **2 = lumen**. The full outer-vessel region is `mask > 0`, and the wall region is defined as the outer-vessel region excluding the lumen.

## Repository structure

| Directory | Purpose |
|---|---|
| [`1_pretraining/`](1_pretraining/) | Self-supervised adaptation of the DINOv3 backbone using ultrasound sweep context |
| [`2_image_model/`](2_image_model/) | Supervised outer-vessel and lumen segmentation on individual annotated images |
| [`3_video_model/`](3_video_model/) | Temporal initialization and training of the final adaptive SCALAR model |
| [`4_inference/`](4_inference/) | YOLO ROI tracking, five-frame segmentation, mask export, and video overlays |

Each module separates model definitions, data handling, losses, and execution logic into focused Python files. The project uses a single root-level dependency list and license.

## Installation

Python **3.11** is recommended. Install a PyTorch build compatible with your operating system and GPU from the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

The published requirements are compatibility ranges, not a frozen record of the original training environment. Exact reproduction additionally requires the original upstream model revisions, pretrained weights, detector implementation, and dataset partitions.

### External components

The repository does **not** redistribute the following:

- [Meta DINOv3](https://github.com/facebookresearch/dinov3) source and original pretrained encoder weights.
- The YOLO vessel detector implementation, configuration, and weights used for ROI localization.
- SCALAR training checkpoints and the corresponding private ROI-cache priors.
- Patient videos, annotations, identifiable records, or clinical linkage tables.

The original detector interface is based on a project-specific `yolo_roi.py` implementation and its utilities. Installing a different YOLO package alone is not necessarily sufficient or equivalent.

## Workflow

The modules follow the sequence **encoder adaptation → image segmentation → temporal segmentation → video inference**. Example paths below are placeholders for authorized, locally held data and weights.

### 1. Self-supervised encoder adaptation

Prepare an approved CSV manifest with a `video_path` column and verify its membership before training. The reference configuration checks for 5,288 unique videos; this check does not independently verify institutional provenance or patient-level exclusions.

```bash
python 1_pretraining/validate_cohort.py \
  --cohort_manifest /private/cohort.csv \
  --video_root /private/videos

python 1_pretraining/pretrain_ssl.py \
  --mode all \
  --cohort_manifest /private/cohort.csv \
  --video_root /private/videos \
  --video_train_file /private/detector_adapter.py \
  --dino_repo /external/dinov3 \
  --dino_weights /private/weights/dinov3_initial.pth \
  --output_dir /private/results/pretraining
```

The detector adapter and training ROI prior should match those used for the original research run. To supply an optional ROI prior, consult the pretraining command-line options (`python 1_pretraining/pretrain_ssl.py --help`). These private files are not stored in the repository.

### 2. Image segmentation

Train the supervised image model using an externally prepared manifest with frozen train/validation/test assignments and LabelMe annotations of the outer boundary and lumen.

```bash
python 2_image_model/train.py \
  --manifest /private/image_manifest.csv \
  --dino-repo /external/dinov3 \
  --dino-weights /private/results/pretraining/dinov3_vitb16_carotid_ssl.pth \
  --output-dir /private/results/image
```

The best image-segmentation checkpoint is used to initialize the temporal segmentation stage.

### 3. Adaptive temporal segmentation

Temporal training has **two stages**. First, a context-aware segmentation model is initialized from the image checkpoint. The final SCALAR model is then initialized from the compatible temporal checkpoint and trained with adaptive feature alignment, reliability estimation, and supervised center/context fusion.

Prepare the five-frame training cache:

```bash
python 3_video_model/train_video.py \
  --mode prepare \
  --manifest /private/video_manifest.csv \
  --yolo_code_dir /private/yolo_detector \
  --video_root /private/videos \
  --output_root /private/results/video
```

Use the available training-stage selectors and initialization-checkpoint options to run the two stages in order:

```bash
python 3_video_model/train_video.py --help
```

**Checkpoint compatibility matters:** the final adaptive model must be initialized from the intended temporal checkpoint, not directly from the image model. Source-level experiment identifiers are retained for compatibility with existing checkpoint metadata; they are not separate public-facing method names.

### 4. Whole-video inference

Run vessel localization, ROI tracking, five-frame SCALAR segmentation, and optional contour-overlay export on an authorized ultrasound video:

```bash
python 4_inference/infer_video.py \
  --video /private/videos/example.mp4 \
  --checkpoint /private/weights/scalar_segmentation.pth \
  --dino-repo /external/dinov3 \
  --dino-weights /private/results/pretraining/dinov3_vitb16_carotid_ssl.pth \
  --yolo-code-dir /private/yolo_detector \
  --training-cache-manifest /private/roi_prior_manifest.csv \
  --output /private/results/inference
```

Use `--video-list` with a CSV containing a `video_path` column for batch inference, or `--no-overlay` to skip generating the visualization video.

**Outputs:**

- `masks_448.npy` — predicted segmentation masks in model ROI coordinates.
- `frame_predictions.csv` — per-frame ROI and inference status.
- `video_done.json` — checkpoint and run metadata.
- `overlay.mp4` — optional contours over the original video frames.

This public inference module does not compute physical-scale measurements, vessel-area trajectories, maximum wall thickness, unfolded thickness maps, or pseudo-3D reconstructions.

## Tests

Run the included unit and synthetic-pipeline tests from the repository root:

```bash
python -m pytest -q \
  1_pretraining/tests \
  2_image_model/tests \
  3_video_model/tests \
  4_inference/tests
```

These tests verify software behavior and input/output interfaces. They do **not** establish clinical accuracy or reproduce the paper's numerical results without the corresponding authorized datasets, pretrained assets, and exact checkpoints.

## Reproducibility and data availability

The repository makes the core learning and inference methods inspectable. The precise results reported in the manuscript depend on approved data partitions, annotation versions, ROI geometry, checkpoint selection, and upstream implementations. These assets are not included here. Users should record model-weight hashes, dependency versions, and dataset versions when reproducing experiments.

Patient information and ultrasound recordings are not distributed with this repository. Any data access requests are subject to the study's ethics approvals, institutional policies, and applicable agreements.

## Citation

If you use this implementation, please cite the SCALAR manuscript when its bibliographic details become available. A formal citation will be added upon publication.

## License

Original SCALAR code that the contributors are authorized to license is provided under the [Apache License 2.0](LICENSE). Third-party software and model weights, including DINOv3 and the selected YOLO implementation, remain subject to their respective terms. The repository license does not grant rights to patient data or to third-party assets.
