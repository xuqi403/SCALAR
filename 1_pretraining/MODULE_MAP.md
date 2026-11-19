# Pretraining module map

The public entry point remains `python 1_pretraining/pretrain_ssl.py`.

| File | Responsibility |
| --- | --- |
| `scalar_ssl/settings.py` | Original SSL/ROI hyperparameters and run-local paths |
| `scalar_ssl/utils.py` | Image, video and reproducibility utilities |
| `scalar_ssl/roi.py` | YOLO detection filtering, coarse and bidirectional ROI tracking |
| `scalar_ssl/cache.py` | Conservative five-frame crop/cache generation and cache validation |
| `scalar_ssl/dataset.py` | SSL clip dataset, weak/strong augmentations and context dropout |
| `scalar_ssl/model.py` | DINOv3 backbone, dense projection heads, student/EMA teacher |
| `scalar_ssl/losses.py` | Dense/global self-distillation and weak contextual prediction loss |
| `scalar_ssl/trainer.py` | Optimization, epoch execution, SSL train/monitor split |
| `scalar_ssl/checkpoints.py` | EMA backbone export and compatibility checks |
| `scalar_ssl/visualization.py` | Representation/nearest-neighbour probes and diagnostic figures |
| `scalar_ssl/runner.py` | CLI and end-to-end orchestration |

Only a private, explicitly approved 5,288-video internal cohort can enter the
paper-default pretraining workflow. No external-centre videos or clinical data
are bundled. The source-only release still needs the original YOLO ROI adapter,
ROI prior manifest for faithful scaling, and official DINOv3 source/checkpoint.
