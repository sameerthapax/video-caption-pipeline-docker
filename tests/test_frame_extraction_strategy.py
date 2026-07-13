from pipeline.extract_frames import compute_dynamic_frame_count


def test_dynamic_frame_count_uses_minimum_for_short_videos():
    assert compute_dynamic_frame_count(5.0, 12) == 6


def test_dynamic_frame_count_scales_up_with_duration():
    assert compute_dynamic_frame_count(90.0, 12) == 9


def test_dynamic_frame_count_caps_at_twelve():
    assert compute_dynamic_frame_count(400.0, 12) == 12


def test_dynamic_frame_count_handles_missing_duration():
    assert compute_dynamic_frame_count(None, 12) == 12


def test_dynamic_frame_count_never_drops_below_six_even_if_max_is_lower():
    assert compute_dynamic_frame_count(20.0, 3) == 6
