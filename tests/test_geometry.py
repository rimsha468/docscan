from itertools import permutations

import numpy as np
import pytest

from docscan.geometry import order_corners, target_size, warp_to_rect

SQUARE = np.float32([[10, 10], [110, 10], [110, 110], [10, 110]])


def signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


@pytest.mark.parametrize("perm", list(permutations(range(4))))
def test_order_ignores_input_order(perm):
    shuffled = SQUARE[list(perm)]
    assert np.allclose(order_corners(shuffled), SQUARE)


def test_order_rotated_quad_is_clockwise_from_top_left():
    angle = np.deg2rad(30)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = (SQUARE - 60) @ rot.T + 200
    ordered = order_corners(pts[::-1])
    assert signed_area(ordered) > 0
    assert ordered[0].sum() == ordered.sum(axis=1).min()


def test_target_size_uses_longest_opposite_edges():
    quad = np.float32([[0, 0], [100, 0], [110, 50], [0, 50]])
    assert target_size(quad) == (110, 51)


def test_warp_returns_expected_shape_and_content():
    image = np.zeros((200, 300, 3), np.uint8)
    image[50:100, 40:140] = (0, 255, 0)
    quad = np.float32([[40, 50], [139, 50], [139, 99], [40, 99]])
    out = warp_to_rect(image, quad)
    assert out.shape[:2] == (49, 99)
    assert out[:, :, 1].mean() > 240


def test_warp_rejects_degenerate_quad():
    image = np.zeros((50, 50, 3), np.uint8)
    quad = np.float32([[5, 5], [5, 5], [5, 5], [5, 5]])
    with pytest.raises(ValueError):
        warp_to_rect(image, quad)
