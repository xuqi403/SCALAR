# Source mapping — D_robust only

Source: user-provided image-training script and optional plotting helper (2026-11-18).

Retained: carotid SSL initialized DINOv3 ViT-B/16, 2/5/8/11 intermediate features, progressive dual-head network, last-four-block fine-tuning, D_robust objective and ultrasound augmentation, topology cleanup, full training/validation and Dice/HD95 evaluation.

Removed deliberately: A_baseline, B_dual, C_dual_hr architectural branches and loss variants; internal clinical Excel enrichment, patient-ID reconciliation, frozen historical split recovery, annotation batch merging, original paths, clinically identifiable audit logs, source-specific Excel reports and paper figure generation.

Only D_robust is represented; the output checkpoint metadata retains `experiment: D_robust`. To reproduce paper results, use original private patient-disjoint split, original carotid SSL weights and verified DINOv3 upstream version.

Do not equate code-level smoke tests with a complete GPU retraining replication.
