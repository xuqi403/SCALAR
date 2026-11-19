"""Single source of truth for the video-training runtime and hyperparameters.

The defaults are copied from the validated V4/V5 release; CLI overrides are set
in this module before preparing caches, training, or evaluating the model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# ---------- Core files ----------
# FINAL clean image hard-case retraining round.  This is the authoritative
# annotation population + split + image baseline for the new video chain.
# Publication-version paths are platform-independent and replace the original
# institution-specific drive mappings. Override them on the command line.
RELEASE_ROOT = Path(__file__).resolve().parent.parent
IMAGE_HARDCASE_RUN_DIR = RELEASE_ROOT / "private"
MANIFEST_PATH = str(RELEASE_ROOT / "private" / "split_manifest.csv")
IMAGE_SPLIT_PATH = MANIFEST_PATH  # Split is already assigned in the supplied manifest.
IMAGE_CHECKPOINT = str(RELEASE_ROOT / "checkpoints" / "image_D_robust.pth")
DINO_REPO = str(RELEASE_ROOT / "external" / "dinov3")
DINO_WEIGHTS = str(RELEASE_ROOT / "checkpoints" / "dinov3_vitb16_carotid_ssl.pth")
PROJECT_ROOT = RELEASE_ROOT / "runs"
VIDEO_RUN_TAG = "paper"
CACHE_DIR = PROJECT_ROOT / "clip_cache"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
VIDEO_ROOTS = []          # Add roots through --video_root if paths are relative.
PATH_REPLACEMENTS = []    # Original institution-specific migrations intentionally removed.
YOLO_CODE_DIR = ""        # Set --yolo_code_dir to the approved detector implementation.
YOLO_CONF_THRESHOLD = 0.40

# ---------- First run ----------
# all      : prepare/reuse cache, then run experiment
# prepare  : only reconstruct clips + YOLO context QC, do not train
# train    : require an existing compatible cache
MODE = "all"

# Start with V0 to verify frame/crop reconstruction; then V1 -> V2 -> V3 -> V4.
DEFAULT_VIDEO_EXPERIMENT = "V5_adaptive_rescue_spur_curv"
RUN_TEST = False
FORCE_REBUILD_CACHE = False

# ---------- Local video clip ----------
# Five frames around the annotated center frame. At ~25 fps this spans ~0.32 s.
# This is intentionally local: enough context to recover an ambiguous boundary,
# but not so long that true longitudinal anatomical change is over-smoothed.
CLIP_OFFSETS = (-4, -2, 0, 2, 4)
CENTER_INDEX = CLIP_OFFSETS.index(0)
IMAGE_SIZE = 448

# Reconstruct the center ROI from raw video and compare it against the saved
# annotation JPG. A very low correlation usually means wrong frame_idx/path/crop.
RECONSTRUCTION_QC = True
RECON_MIN_CORRELATION = 0.70
RECON_MAX_MEAN_ABS_DIFF = 55.0

# Context-frame YOLO is a reliability cue, not a hard requirement. The fixed center
# crop is still available when YOLO momentarily fails.
CONTEXT_WEIGHT_GOOD_MIN = 0.50
CONTEXT_WEIGHT_MISSING = 0.30
CONTEXT_WEIGHT_UNSTABLE = 0.15
CONTEXT_WEIGHT_OUT_OF_RANGE = 0.0

# ---------- Context ROI tracking ----------
# fixed_anchor:
#     every frame uses the center annotation crop (original baseline)
# yolo_track:
#     neighboring frames are spatially aligned using YOLO center shift while
#     preserving the original crop size
# hybrid:
#     YOLO correction is used only when the detected box passes QC
CONTEXT_ROI_MODE = "yolo_track"

# Anchor-relative geometry QC. The anchor is the known center annotation crop.
CONTEXT_MIN_IOU = 0.02
CONTEXT_MAX_CENTER_SHIFT_IMAGE_DIAG = 0.20
CONTEXT_MIN_SIZE_RATIO = 0.30
CONTEXT_MAX_SIZE_RATIO = 3.00

# ---------- Training ----------
SEED = 42
BATCH_SIZE = 2                 # 5 x 448 DINO frames/sample; safe start for dual 4090
NUM_WORKERS = 4
EPOCHS = 20
WEIGHT_DECAY = 1e-4
LR_DECODER = 1e-4
LR_TEMPORAL = 1e-4
LR_BACKBONE = 5e-6
AMP = True
PRED_THRESHOLD = 0.50

# Same checkpoint selection rule as image segmentation.
CHECKPOINT_OUTER_WEIGHT = 0.25
CHECKPOINT_WALL_WEIGHT = 0.50
CHECKPOINT_LUMEN_WEIGHT = 0.25

# Current image D model loss components retained for the video model.
LAMBDA_WALL = 1.0
LAMBDA_CONTAINMENT = 0.10
LAMBDA_BOUNDARY = 0.25

# ---------- First-round OUTER protrusion regularization ----------
# Suppress thin/long outward "feet" and spikes without imposing a circle,
# ellipse, convexity, or global smoothness prior.
#
# Manual single-frame outer boundaries are not treated as infinitely precise:
# no penalty is applied within a small tolerance band around GT. Beyond that,
# the penalty increases in coarse distance bands.
LAMBDA_OUTER_SPUR = 0.15
OUTER_SPUR_ALLOW_PX = 4
OUTER_SPUR_MID_PX = 8
OUTER_SPUR_FAR_PX = 16
OUTER_SPUR_PROB_POWER = 2.0

# ---------- Second-round weak local OUTER curvature regularization ----------
# This is prediction-only and does NOT force the contour to match a circle,
# ellipse, convex hull, or the small polygonal irregularities of the manual GT.
#
# It penalizes only abrupt pixel-scale changes in the local boundary normal.
# Slow anatomical curvature (CCA, bulb, bifurcation, eccentric plaque) is allowed.
#
# LAMBDA_OUTER_CURV is deliberately small. The target is for the weighted
# curvature contribution to remain a minor regularizer, not a dominant loss.
LAMBDA_OUTER_CURV = 0.02
OUTER_CURV_DEADZONE = 0.15
OUTER_CURV_SMOOTH_KERNEL = 3
OUTER_CURV_GRAD_SCALE = 0.02

# Temporal regularization (used in V3/V4 only).
LAMBDA_AUX_CENTER = 0.20
LAMBDA_TEMPORAL_AREA = 0.05
LAMBDA_TEMPORAL_CENTROID = 0.02

# V4 only: decoder/temporal warm-up, then fine-tune last 4 DINO blocks.
V4_BACKBONE_WARMUP_EPOCHS = 3

# V5 adaptive rescue only. These are training/architecture settings, NOT inference
# thresholds deciding how much the model must trust center vs context. The final
# center/context proportion is learned as a spatial gate G(x,y).
V5_INIT_CHECKPOINT = (
    EXPERIMENT_ROOT / f"V4_attn_last4__{VIDEO_RUN_TAG}" / "best_video_checkpoint.pth"
)
# IMPORTANT: freeze Round 1 V4 initialization when starting a new Round 2 V5
# experiment. Do NOT change this to a nonexistent V4__hardcase_round2 path.
V5_ADAPTIVE_WARMUP_EPOCHS = 3
V5_BACKBONE_WARMUP_EPOCHS = 8
V5_MAX_ALIGN_OFFSET_FEATURE_PX = 2.0
V5_CENTER_CORRUPT_PROB = 0.55
V5_CONTEXT_DROPOUT_PROB = 0.35
V5_GATE_TEMPERATURE = 0.35
LAMBDA_V5_CENTER_AUX = 0.10
LAMBDA_V5_CONTEXT_AUX = 0.20
LAMBDA_V5_GATE = 0.10

# ---------- Topology ----------
TOPOLOGY_MIN_COMPONENT_PX = 24
TOPOLOGY_LUMEN_OUTER_OVERLAP_MIN = 0.75
TOPOLOGY_OUTER_SMOOTH = True
TOPOLOGY_SMOOTH_SIGMA = 0.8
TOPOLOGY_SMOOTH_MAX_AREA_CHANGE = 0.02
TOPOLOGY_SMOOTH_MIN_IOU = 0.97

# ---------- QC-triggered temporal rescue ----------
# Evaluated after the temporal model is trained. This does NOT train a separate model.
QC_CONFIDENCE_MIN = 0.72
QC_OUTER_AREA_JUMP_MAX = 0.35
QC_LUMEN_AREA_JUMP_MAX = 0.40
QC_RAW_LUMEN_OUTSIDE_MAX_PX = 16

# ---------- Monitoring ----------
FIXED_PREVIEW_N = 20
PREVIEW_EVERY = 2
RANDOM_PREVIEW_N = 20
EARLY_STOP_PATIENCE = 0      # 0 = disabled for V1-V3; V4 can be stopped manually

# Save one model-only checkpoint for EVERY epoch in addition to the three
# independently selected candidates below. This is intentionally enabled here
# because small validation-metric differences may coexist with annotation noise.
# No optimizer/scaler state is stored, keeping each file smaller.
SAVE_EVERY_EPOCH_CHECKPOINT = True

# ---------- Temporal stability evaluation ----------
# Evaluation-only. Does NOT change training, loss, checkpoint selection, or cache.
# For each cached 5-frame clip, the trained video model is re-centered on each
# of the 5 frames and produces a dense local prediction sequence. The frozen
# image model predicts the same 5 frames independently. This enables a fair
# paired comparison of temporal stability without requiring dense manual masks.
RUN_TEMPORAL_STABILITY_EVAL = True

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


VIDEO_EXPERIMENTS = {
    "V4_attn_last4": dict(
        description="Attention fusion + temporal anatomy regularization; last-4 backbone tuning",
        fusion="attention", train=True, temporal_loss=True, fine_tune_last4=True,
        adaptive_rescue=False, outer_spur_loss=False, outer_curvature_loss=False,
    ),
    "V5_adaptive_rescue_spur_curv": dict(
        description="SCALAR: feature alignment, neighbour reliability, relative-error gate supervision, outer spur/curvature regularization",
        fusion="adaptive", train=True, temporal_loss=False, fine_tune_last4=True,
        adaptive_rescue=True, outer_spur_loss=True, outer_curvature_loss=True,
    ),
}


CSV_EXTRA_ENCODINGS = ()
CACHE_SCHEMA_VERSION = 2
