"""Module-boundary smoke tests for the public inference package."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scalar_infer import io_utils, roi, feature_bank, segmentation, overlay, pipeline, model_loader

def test_module_boundaries():
    assert callable(roi.detect_video)
    assert callable(roi.build_roi_track)
    assert callable(feature_bank.extract_feature_bank)
    assert callable(segmentation.infer_feature_windows)
    assert callable(overlay.write_overlay_video)
    assert callable(model_loader.load_v5_model)
    assert callable(pipeline.run_one)

def test_no_physical_measurement_modules():
    root = Path(__file__).resolve().parents[1]
    assert not (root / 'physical_measurements.py').exists()
    assert not (root / 'scalar_infer' / 'physical_measurements.py').exists()
    src = '\n'.join(p.read_text(encoding='utf-8') for p in (root/'scalar_infer').glob('*.py'))
    forbidden = ('mm_per_pixel', 'ruler_pixels', 'wall_thickness_map', 'sequence_areas', 'make_physical_reconstruction')
    assert not any(term in src for term in forbidden)

def test_exposes_batch_controls():
    import inspect
    assert 'batch_size' in inspect.signature(feature_bank.extract_feature_bank).parameters
    assert 'fusion_batch_size' in inspect.signature(segmentation.infer_feature_windows).parameters
