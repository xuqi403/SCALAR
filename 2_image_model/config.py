"""Final D_robust image-segmentation model: configuration used by SCALAR."""

MODEL_CONFIG = dict(
    name="D_robust",
    description="Dual outer/lumen segmentation with progressive decoder, ultrasound augmentation, boundary supervision and last four DINOv3 blocks",
    model="dual", decoder="progressive", loss="dual", augmentation="ultrasound",
    freeze_epochs=3, unfreeze="last4", lr_decoder=1e-4, lr_backbone=5e-6,
    lambda_wall=1.0, lambda_containment=0.10, lambda_boundary=0.25,
    early_stop=5,
)
IMAGE_SIZE = 448
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 15
SEED = 42
WEIGHT_DECAY = 1e-4
PRED_THRESHOLD = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_SCORE_WEIGHTS = {"outer": 0.25, "wall": 0.50, "lumen": 0.25}
