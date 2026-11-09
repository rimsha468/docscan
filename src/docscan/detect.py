import cv2
import numpy as np

from .geometry import order_corners


class DocumentNotFound(Exception):
    pass


def _candidate_masks(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    yield edges

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    yield cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    yield cv2.morphologyEx(cv2.bitwise_not(bright), cv2.MORPH_CLOSE, kernel)


def _quad_from_contour(contour):
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for ratio in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, ratio * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def find_document(image, work_height=500, min_area=0.2):
    """Locate the largest four-sided region and return its corners.

    Corners come back in the coordinates of the original image, ordered
    top-left, top-right, bottom-right, bottom-left.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    scale = min(1.0, work_height / height)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    frame_area = gray.shape[0] * gray.shape[1]
    for mask in _candidate_masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area * frame_area:
                break
            if area > 0.98 * frame_area:
                continue
            quad = _quad_from_contour(contour)
            if quad is not None:
                return order_corners(quad / scale)

    raise DocumentNotFound("no four-sided document outline found")
