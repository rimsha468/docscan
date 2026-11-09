import numpy as np
import pytest

from docscan import draw_outline, scan
from tests.synthetic import make_scene


def test_color_mode_returns_three_channels():
    photo, _ = make_scene()
    result = scan(photo, "color")
    assert result.image.ndim == 3
    assert result.corners.shape == (4, 2)


def test_gray_mode_is_single_channel():
    photo, _ = make_scene()
    assert scan(photo, "gray").image.ndim == 2


def test_bw_mode_is_strictly_binary():
    photo, _ = make_scene()
    out = scan(photo, "bw").image
    assert out.ndim == 2
    assert set(np.unique(out)) <= {0, 255}


def test_unknown_mode_is_rejected():
    photo, _ = make_scene()
    with pytest.raises(ValueError):
        scan(photo, "sepia")


def test_bw_removes_shadow_gradient():
    photo, _ = make_scene(shadow=True)
    out = scan(photo, "bw").image
    w = out.shape[1]
    left_margin = out[:, 8:25]
    right_margin = out[:, w - 25 : w - 8]
    assert (left_margin == 255).mean() > 0.95
    assert (right_margin == 255).mean() > 0.95
    assert 0.005 < (out == 0).mean() < 0.15


def test_outline_does_not_modify_input():
    photo, _ = make_scene()
    before = photo.copy()
    result = scan(photo)
    drawn = draw_outline(photo, result.corners)
    assert np.array_equal(photo, before)
    assert drawn.shape == photo.shape
    assert not np.array_equal(drawn, photo)


def test_trim_shrinks_output_and_zero_keeps_full_warp():
    photo, _ = make_scene()
    full = scan(photo, trim=0).image
    trimmed = scan(photo, trim=0.05).image
    assert trimmed.shape[0] < full.shape[0]
    assert trimmed.shape[1] < full.shape[1]
