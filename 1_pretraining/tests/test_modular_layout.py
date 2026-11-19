"""Regression checks for the modularized SSL code and shared runtime settings."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalar_ssl import cache, dataset, losses, model, roi, runner, settings, trainer, utils


def test_modules_preserve_core_configuration():
    assert settings.EXPECTED_INTERNAL_VIDEO_COUNT == 5288
    assert settings.CONTEXT_OFFSETS == (-4, -2, 0, 2, 4)
    assert settings.SSL_LAYERS == [8, 9, 10, 11]
    assert settings.TRAINABLE_LAST_N_BLOCKS == 4
    assert settings.EPOCHS == 3
    assert settings.CACHE_SCHEMA_VERSION == 3
    assert callable(runner.main)
    assert callable(trainer.run_ssl_epoch)
    assert callable(losses.context_prediction_loss)
    assert callable(cache.build_roi_cache)
    assert callable(roi.bidirectional_local_track)
    assert callable(dataset.make_ssl_clip_views)
    assert callable(model.load_dinov3)


def test_mutable_runtime_paths_are_shared_between_modules(tmp_path, monkeypatch):
    video = tmp_path / "private_test_video.mp4"
    video.write_bytes(b"placeholder only; no patient data")
    monkeypatch.setattr(settings, "VIDEO_ROOTS", [tmp_path])
    monkeypatch.setattr(settings, "OUTPUT_ROOT", tmp_path / "output")
    assert utils.resolve_video_path(video.name) == video.resolve()
    assert cache.settings.OUTPUT_ROOT == tmp_path / "output"
    assert runner.settings.OUTPUT_ROOT == tmp_path / "output"
