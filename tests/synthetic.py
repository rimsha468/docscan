"""Builds a fake photo of a page on a table so detection can be tested offline.

Run `python -m tests.synthetic` to regenerate the images used in the README.
"""
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

LINES = [
    "Revenue grew in the second half of the year,",
    "driven mostly by renewals from existing clients.",
    "Support tickets fell by a fifth after the new",
    "onboarding guide shipped in March.",
    "Hiring stays on plan: four engineers and one",
    "designer are due to start before the summer.",
    "The main risk is the delayed warehouse move,",
    "which pushes two product launches into autumn.",
    "Budget requests for next year are listed on",
    "the following page, grouped by department.",
    "Questions can go to the finance team any time",
    "before the review meeting on the fourteenth.",
]

DEFAULT_CORNERS = ((230, 120), (900, 170), (950, 880), (190, 820))


def make_page(width=600, height=800):
    page = np.full((height, width, 3), 242, np.uint8)
    ink = (35, 35, 35)
    cv2.putText(page, "QUARTERLY REPORT", (40, 90), FONT, 1.3, ink, 3, cv2.LINE_AA)
    cv2.line(page, (40, 118), (width - 40, 118), ink, 2)
    y = 175
    for line in LINES:
        cv2.putText(page, line, (40, y), FONT, 0.6, ink, 1, cv2.LINE_AA)
        y += 42
    return page


def make_scene(corners=DEFAULT_CORNERS, size=(1200, 1000), shadow=True, seed=0):
    """Return (photo, corners) with the page placed on a noisy brown table."""
    rng = np.random.default_rng(seed)
    width, height = size

    table = np.empty((height, width, 3), np.float32)
    table[:] = (55, 85, 125)
    table *= np.linspace(0.8, 1.15, height, dtype=np.float32)[:, None, None]
    table += rng.normal(0, 10, table.shape).astype(np.float32)
    for x in range(0, width, 37):
        table[:, x : x + 2] *= 0.92

    page = make_page().astype(np.float32)
    ph, pw = page.shape[:2]
    if shadow:
        page *= np.linspace(1.0, 0.6, pw, dtype=np.float32)[None, :, None]

    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    dst = np.float32(corners)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, matrix, (width, height))
    mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), matrix, (width, height))

    photo = table * (1 - mask[..., None]) + warped * mask[..., None]
    photo += rng.normal(0, 2, photo.shape).astype(np.float32)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    return np.clip(photo, 0, 255).astype(np.uint8), np.float32(corners)


if __name__ == "__main__":
    from docscan import draw_outline, scan

    out = Path(__file__).resolve().parent.parent / "docs"
    out.mkdir(exist_ok=True)
    photo, _ = make_scene()
    cv2.imwrite(str(out / "sample_photo.jpg"), photo)
    result = scan(photo, "bw")
    cv2.imwrite(str(out / "sample_outline.jpg"), draw_outline(photo, result.corners))
    cv2.imwrite(str(out / "sample_scan.png"), result.image)
    print("wrote", sorted(p.name for p in out.iterdir()))
