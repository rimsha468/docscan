import cv2
import numpy as np
import pytest

from docscan import DocumentNotFound, find_document
from tests.synthetic import make_scene

LAYOUTS = [
    ((230, 120), (900, 170), (950, 880), (190, 820)),
    ((150, 200), (1000, 150), (1050, 800), (200, 900)),
    ((300, 100), (800, 120), (820, 900), (280, 880)),
    ((250, 160), (950, 300), (880, 900), (180, 760)),
]


@pytest.mark.parametrize("corners", LAYOUTS)
def test_finds_corners_on_synthetic_photo(corners):
    photo, truth = make_scene(corners)
    found = find_document(photo)
    assert found.shape == (4, 2)
    assert np.linalg.norm(found - truth, axis=1).max() < 12


def test_accepts_grayscale_input():
    photo, truth = make_scene()
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    found = find_document(gray)
    assert np.linalg.norm(found - truth, axis=1).max() < 12


def test_works_without_shadow():
    photo, truth = make_scene(shadow=False, seed=3)
    found = find_document(photo)
    assert np.linalg.norm(found - truth, axis=1).max() < 12


def test_raises_on_blank_image():
    blank = np.full((600, 800, 3), 120, np.uint8)
    with pytest.raises(DocumentNotFound):
        find_document(blank)


def test_ignores_regions_below_min_area():
    frame = np.full((600, 800, 3), 40, np.uint8)
    cv2.rectangle(frame, (300, 250), (420, 350), (230, 230, 230), -1)
    with pytest.raises(DocumentNotFound):
        find_document(frame)
