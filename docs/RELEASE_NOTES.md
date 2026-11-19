# Change log / fidelity statement

Input: 2,231-line original user-supplied SSL script dated 2026-11-18.

Retained verbatim in behavior: original box tracking, sampling, ROI crops, 5-frame SSL dataset,
augmentations, DINOv3 network and loss modules, EMA optimization, validation/split logic,
checkpoint export and compatibility verification.

Changed:
1. Replaced hospital-machine absolute paths with explicit command-line paths.
2. Replaced annual-table aggregation with a single validated, private, internal-only
   exact-5,288-video manifest.
3. Added startup checks, train-mode stale-cache rejection and private provenance manifests.
4. Changed user-facing downstream print hints to avoid referring to a local V5 file.
5. Added documentation, dependencies, .gitignore and cohort/component tests.

Not changed: original 3-epoch, 448px, 5-context, four-trainable-block, EMA and loss settings.

Not independently verified: ROI detector code and weights, DINOv3 checkpoint/loading,
original ROI prior, real training outputs, exact patient/video provenance and full GPU run.

## Modular inference refactor
`4_inference/core.py` replaced with a `scalar_infer/` package. Command and outputs unchanged; no measurements or Figure 5 utilities included.
