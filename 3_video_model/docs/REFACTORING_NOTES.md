# Modular refactor traceability

Source: `SCALAR_video_model_V4_V5_GitHub_ready.zip` (pre-modular public candidate).

This change relocates complete functions and classes into topic modules without
rewriting their internal algorithms. Runtime constants are read from a shared
`scalar_video.settings` module. The single-file `train_video.py` becomes a
backward-compatible import/CLI shim.

One variable rename (`settings` → `fingerprint_settings`) was necessary in
`cache_fingerprint` to avoid shadowing the shared settings module. The hashed
contents and fingerprint computation are unchanged.

The V4→V5 training initialization, selection score, checkpoint filenames,
`CLIP_OFFSETS=(-4,-2,0,2,4)`, and adaptive gate temperature 0.35 are retained.

Validation performed without private data:
- Original and modular V4: identical state-dict tensors, forward tensors,
  total anatomy loss, and temporal regularization loss on fixed mock inputs.
- Original and modular V5: identical state-dict tensors, forward tensors,
  total anatomy loss, and adaptive rescue supervision loss on fixed mock inputs.
- All bundled Python offline tests pass.

Does not establish numerical equivalence on clinical videos until the original
weights, data, YOLO detector and CUDA environment are used.
