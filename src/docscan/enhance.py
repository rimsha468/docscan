import cv2


def _odd(n):
    return n if n % 2 else n + 1


def flatten_lighting(gray):
    """Divide out the slowly varying background so shadows and gradients drop away."""
    size = _odd(max(3, min(gray.shape[:2]) // 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    background = cv2.dilate(gray, kernel)
    background = cv2.medianBlur(background, _odd(size * 3))
    return cv2.divide(gray, background, scale=255)


def apply_mode(image, mode):
    if mode == "color":
        return image
    if mode not in ("gray", "bw"):
        raise ValueError(f"unknown mode: {mode!r}")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = flatten_lighting(gray)
    if mode == "gray":
        return flat

    block = _odd(max(15, min(flat.shape[:2]) // 25))
    return cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15
    )
