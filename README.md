# SCALAR

**Adaptive cross-sectional context enables dense vessel segmentation in sparsely annotated carotid ultrasound sweeps**

SCALAR is a research framework for segmenting the outer-vessel boundary and lumen from routine transverse carotid B-mode ultrasound sweeps. It uses a carotid-adapted DINOv3 image encoder and neighbouring cross-sections to infer the anatomy of a target frame. This repository provides the four essential components of the segmentation workflow, organized for inspection and reuse.

> **Research use only.** This software has not been validated as a medical device and is not intended for clinical diagnosis, treatment, or patient management.

## Repository structure

```text
SCALAR/
├── README.md                 # This project-wide guide
├── requirements.txt          # Unified Python dependencies
├── LICENSE                   # License for eligible SCALAR-authored code
├── THIRD_PARTY_NOTICES.md    # Third-party code and weights
├── .gitignore
│
├── 1_pretraining/
│   ├── pretrain_ssl.py       # DINOv3 carotid sweep-context SSL
│   ├── cohort.py             # Approved-cohort validation
│   └── validate_cohort.py
│
├── 2_image_model/
│   ├── model.py              # D_robust image segmentation architecture
│   ├── dataset.py            # Labelme masks and image preprocessing
│   ├── losses.py
│   ├── train.py
│   ├── postprocess.py
│   └── metrics.py
│
├── 3_video_model/
│   ├── train_video.py        # V4/V5 training entry point
│   ├── plot.py               # Optional training diagnostics
│   └── scalar_video/        # Data, cache, models, losses and trainer
│
└── 4_inference/
    ├── infer_video.py        # Whole-video segmentation entry point
    └── scalar_infer/        # ROI, feature bank, fusion and outputs
```

Tests, non-patient templates, and implementation notes are included alongside the corresponding modules. The four numbered directories deliberately have no separate dependency or license declarations; this root directory is authoritative.

## Method at a glance

1. **Self-supervised pretraining.** Adapt the DINOv3 ViT-B/16 backbone to carotid ultrasound sweeps using an EMA teacher and spatially aligned five-frame context.
2. **Image segmentation.** Train the final `D_robust` progressive dual-head outer/lumen model from the carotid SSL backbone. The wall region is derived from the outer and lumen masks.
3. **Video segmentation.** Train `V4_attn_last4` from the image checkpoint, then initialize final `V5_adaptive_rescue_spur_curv` from the compatible V4 checkpoint. V5 adds learned local alignment, neighbour reliability and an adaptive centre/context rescue gate with supervised selection.
4. **Whole-video inference.** Run YOLO carotid ROI detection/tracking, DINOv3 feature extraction, five-frame V5 fusion and native-frame segmentation overlays.

Model input size is **448 × 448**. Video context uses frame offsets **(-4, -2, 0, +2, +4)**. The pixel-label convention is **0 = background, 1 = vessel wall, 2 = lumen**; the complete outer-vessel mask is `label > 0`.

### Scope of this public code

Included: the final SSL, `D_robust`, V4 and V5 training implementations; necessary data interfaces; video segmentation inference; basic training-time metrics, postprocessing and diagnostics.

Not included: patient videos or annotations, clinical spreadsheets, original patient-level cohort allocations, model checkpoints, the upstream DINOv3 or YOLO source/weights, V0–V3 exploratory training scripts, full manuscript plotting/evaluation pipelines, automatic image-scale calibration, mm/mm² measurements, area profiles, maximum wall thickness, unfolded thickness maps or Figure 5 reconstruction.

## Installation

Python **3.11** is recommended. Install a PyTorch/torchvision build matching the available CPU/GPU and CUDA runtime (see the [official PyTorch installer](https://pytorch.org/get-started/locally/)), then install this repository's common dependencies:

```bash
conda create -n scalar python=3.11 -y
conda activate scalar
# Install matching torch and torchvision for your hardware first.
python -m pip install -r requirements.txt
```

The version ranges in `requirements.txt` are **compatibility ranges**, not a lockfile of the original research environment. For closer reproduction, record the original PyTorch/CUDA versions, the exact DINOv3 source revision, detector revision, and SHA-256 hashes of all trained checkpoints. Independently install the dependencies required by the official DINOv3 repository and by the specific YOLO detector implementation used in your environment.

### External dependencies and model files

The following are required for actual training/inference but are **not** contained in this source repository:

| Asset | Used by | Notes |
|---|---|---|
| Official [Meta DINOv3](https://github.com/facebookresearch/dinov3) source and original ViT-B/16 weights | Pretraining | Subject to Meta's DINOv3 License |
| SCALAR carotid SSL backbone checkpoint | Image/video model and inference | Produced by module 1 |
| Final `D_robust` image checkpoint | V4 training | Produced by module 2 |
| Compatible `V4_attn_last4` checkpoint | V5 training | **V5 is initialized from V4, not directly from the image model** |
| Final V5 checkpoint (`best_anatomy_checkpoint.pth` for whole-video inference) | Inference | Must match the exact manuscript checkpoint |
| Compatible YOLO ROI detector source, configuration and checkpoint | ROI preparation and inference | Must match the original training/ROI geometry |
| Training ROI-cache manifest/NPZ anchors, where available | Faithful ROI scale reconstruction | Private; absence activates a fallback that may change outputs |

The original YOLO adapter expects a `yolo_roi.py` module (and `utils/utils.py`) and the detector weights configured by that code. This is **not** automatically interchangeable with `pip install ultralytics`; confirm the exact detector implementation before use.

## Input conventions and examples

Input paths below are placeholders, not downloadable datasets. Keep private manifests and outputs outside the Git repository. Do **not** publish patient identifiers, metadata linkage tables, ultrasound images, annotation JSONs or institution-specific directory structures.

**Pretraining cohort:** Module 1 accepts an explicitly approved `video_path` CSV containing **5,288 unique, internal-centre videos**. This reflects the final specified pretraining cohort; the code checks count/path uniqueness but cannot independently establish institutional provenance. The user must exclude any held-out or otherwise ineligible videos/patients according to the final study protocol.

**Image annotations:** Module 2 expects a CSV with `split`, `annotation_image_path`, `annotation_json_path` and preferably a non-identifying `group_id`. The Labelme JSON must contain `wall` (outer-boundary polygon) and `lumen` polygons. Train/validation/test assignments must be prepared privately; the released code does not re-create the original cohort allocation.

**Video annotations:** Module 3 accepts either the original five-frame NPZ cache plus manifest or a frozen-split CSV describing the source video, centre frame, Labelme files, and the original annotation ROI coordinates. The original split must agree with the image-model split.

### 1. Prepare the DINOv3 pretraining cohort and train SSL

```bash
python 1_pretraining/validate_cohort.py \
  --cohort_manifest /private/internal_5288.csv \
  --video_root /private/videos \
  --excluded_manifest /private/exclusions.csv

python 1_pretraining/pretrain_ssl.py \
  --mode all \
  --cohort_manifest /private/internal_5288.csv \
  --video_root /private/videos \
  --excluded_manifest /private/exclusions.csv \
  --video_train_file /private/original_detector_adapter/train_v5.py \
  --v5_cache_manifest /private/original_video_clip_manifest.csv \
  --dino_repo /licensed/dinov3 \
  --dino_weights /private/weights/original_dinov3_vitb16.pth \
  --output_dir /private/results/ssl
```

The original detector adapter and ROI prior are dependencies of the specified pretraining procedure. A different detector or fallback crop strategy may be useful for demonstration but is not an exact reproduction of the manuscript training run.

### 2. Train the image model (`D_robust`)

```bash
python 2_image_model/train.py \
  --manifest /private/image_split.csv \
  --dino-repo /licensed/dinov3 \
  --dino-weights /private/results/ssl/dinov3_vitb16_carotid_ssl.pth \
  --output-dir /private/results/image
```

This produces an image segmentation checkpoint such as `best_checkpoint.pth`. The older A/B/C exploratory configurations are not included.

### 3. Train the video model (V4, then V5)

Prepare the private five-frame cache once (skip this step if a compatible original cache is already available):

```bash
python 3_video_model/train_video.py \
  --mode prepare \
  --manifest /private/video_split.csv \
  --yolo_code_dir /private/yolo_detector \
  --video_root /private/videos \
  --output_root /private/results/video
```

Train V4 with the final image checkpoint:

```bash
python 3_video_model/train_video.py \
  --mode train --experiment V4_attn_last4 \
  --cache_dir /private/results/video/clip_cache \
  --image_checkpoint /private/results/image/best_checkpoint.pth \
  --dinov3_repo /licensed/dinov3 \
  --dinov3_weights /private/results/ssl/dinov3_vitb16_carotid_ssl.pth \
  --output_root /private/results/video
```

Then train the final V5 using the **exact compatible V4 checkpoint**:

```bash
python 3_video_model/train_video.py \
  --mode train --experiment V5_adaptive_rescue_spur_curv \
  --cache_dir /private/results/video/clip_cache \
  --image_checkpoint /private/results/image/best_checkpoint.pth \
  --dinov3_repo /licensed/dinov3 \
  --dinov3_weights /private/results/ssl/dinov3_vitb16_carotid_ssl.pth \
  --v4_checkpoint /private/weights/paper_V4_checkpoint.pth \
  --output_root /private/results/video
```

**Important:** The original final V5 experiment used a specific earlier V4 initialization checkpoint. Do not assume that retraining V4 on a newly prepared split is identical to the original experiment. Record the actual V4 and V5 checkpoint hashes before publishing manuscript-specific reproduction commands.

### 4. Infer a video

```bash
python 4_inference/infer_video.py \
  --video /private/videos/example.mp4 \
  --checkpoint /private/weights/best_anatomy_checkpoint.pth \
  --dino-repo /licensed/dinov3 \
  --dino-weights /private/results/ssl/dinov3_vitb16_carotid_ssl.pth \
  --yolo-code-dir /private/yolo_detector \
  --training-cache-manifest /private/results/video/clip_cache/video_clip_manifest.csv \
  --output /private/results/inference
```

Alternatively use `--video-list /private/videos.csv` containing a `video_path` column; add `--no-overlay` to omit MP4 visualization. The default model definition is loaded from `3_video_model/train_video.py`.

Per-video outputs are `masks_448.npy` (model-ROI coordinates), `frame_predictions.csv` (ROI and frame status), `video_done.json` (run/checkpoint metadata), and optional `overlay.mp4` (contours in native video coordinates). No physical-scale or wall-thickness measurements are computed.

## Tests

The source repository includes unit tests and synthetic/mock pipeline tests:

```bash
python -m pytest -q 1_pretraining/tests 2_image_model/tests 3_video_model/tests 4_inference/tests
```

These tests check code paths and basic input/output conventions; they do **not** establish paper-level accuracy or verify the full workflow with the original licensed assets. A complete real-weight, real-video reproduction test is still needed before declaring the code release validated.

## Reproducibility and limitations

- The repository is intentionally restricted to the core learning and segmentation algorithm; private data preparation, patient-level split audits, external validation processing and publication-only figures are not released here.
- Manuscript results require the exact approved data and model/checkpoint versions, including the distinction between `best_dice` and `best_anatomy` selections. Do not claim that the public code alone reproduces the paper's numerical results without those assets.
- For mm-based boundary accuracy, thickness and Figure 5 results, consult the manuscript methods; those measurement/visualization implementations are outside the scope of this repository and must be retained by the authors for scientific verification where required.
- Respect patient confidentiality, third-party model licenses and institutional permissions.

## Citation and licensing

When a final manuscript citation or DOI is available, add it here and cite the SCALAR paper when using this research software.

The **original SCALAR-authored portions** are proposed for release under the [Apache License 2.0](LICENSE), subject to confirmation by all relevant copyright owners and institutions **before public release**. This license does not override the separate DINOv3 License, the license of the particular YOLO implementation, or any model-weight/data distribution terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No source, weights or patient data owned by others are relicensed by this repository.
