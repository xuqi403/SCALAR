# SCALAR pretraining — dependency collection and release audit

## Please provide to complete exact method reproduction

1. **`train_v5.py`** (the final SCALAR V5 train module used by the uploaded SSL script) and any of its custom YOLO modules or weights where distribution is allowed. This supplies:
   - `import_yolo_components()` returning `(YOLO, cvtColor, preprocess_input, resize_image)`;
   - `yolo_detect_boxes(yolo, pil_image, cvtColor, preprocess_input, resize_image)` returning a list of dicts with `x1,y1,x2,y2,score`.
   The SSL package has NOT reimplemented or substituted this detector.
2. **Original `video_clip_manifest.csv`**: needed to match ROI-size prior from image-model training. Must include `split`, `cache_path`, `video_width`, `video_height`; `cache_path` refers to private NPZ file(s) containing `anchor_box`.
3. **DINOv3 repository version/commit and original checkpoint identifier/hash**, to pin upstream implementation and seed weights. These downloads do not need to go on GitHub.
4. **Private exact 5,288-video list** and evidence of exclusions of downstream labelled and held-out dense videos/patients. Do not send the identifiable list to a public repository; you can run the validator locally and share count-only summaries.
5. **Original environment record** (`pip freeze` or conda export, GPU/CUDA/PyTorch versions) and confirmed paper checkpoint hashes, if available.

## Confirmations necessary before public release

- Verify 5,288 is *eligible videos* and report actual successful ROI cache / SSL training and monitoring counts separately.
- Confirm all 5,288 were drawn from internal center records and that no external-center video entered any cache.
- Confirm 40 dense held-out videos excluded **before SSL** (and any additional labelled cohort exclusions as documented in manuscript).
- Compare the resulting SSL checkpoint with the checkpoint that initialized the manuscript image and video models.
- Remove patient identifiers, directory paths, frame previews, private manifests and checkpoint files from Git before publishing.
- Verify third-party licensing for DINOv3 and any YOLO implementation/checkpoints. Apache-2.0 here applies only to original authors' code.
