from pipeline.extract_frames import compute_dynamic_frame_count


def test_dynamic_frame_count_respects_cap_for_short_videos():
    assert compute_dynamic_frame_count(20.0, 12) == 10


def test_dynamic_frame_count_respects_cap_for_longer_videos():
    assert compute_dynamic_frame_count(75.0, 12) == 12


def test_dynamic_frame_count_handles_missing_duration():
    assert compute_dynamic_frame_count(None, 12) == 12


def test_dynamic_frame_count_default_cap_is_three():
    assert compute_dynamic_frame_count(120.0, 3) == 3
