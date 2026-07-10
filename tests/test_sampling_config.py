from pipeline.run_extraction_stage import build_sampling_config
from pipeline.temporal_segments import build_segment_boundaries


def test_sampling_config_uses_lower_frame_counts_for_short_clips():
    config = build_sampling_config(8.0)

    assert config.scene_change_count >= 1
    assert config.uniform_count >= 1
    assert config.safety_count >= 1
    assert config.scene_change_count + config.uniform_count + config.safety_count <= 8


def test_sampling_config_scales_up_without_exceeding_cap_for_longer_clips():
    config = build_sampling_config(60.0)

    assert config.scene_change_count >= config.uniform_count
    assert config.scene_change_count + config.uniform_count + config.safety_count <= 12


def test_temporal_segments_default_to_three_equal_ranges():
    boundaries = build_segment_boundaries(30.0)

    assert len(boundaries) == 3
    assert boundaries[0] == (0.0, 10.0, "0-33")
    assert boundaries[1] == (10.0, 20.0, "33-67")
    assert boundaries[2] == (20.0, 30.0, "67-100")
