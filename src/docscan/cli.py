import argparse
import sys
from pathlib import Path

import cv2

from .detect import DocumentNotFound
from .pipeline import draw_outline, scan


def build_parser():
    parser = argparse.ArgumentParser(
        prog="docscan",
        description="Flatten photographed documents into top-down scans.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="input photos")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["color", "gray", "bw"],
        default="color",
        help="output style (default: color)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("scans"),
        help="where to write results (default: ./scans)",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.01,
        help="fraction of each side to crop after warping (default: 0.01)",
    )
    parser.add_argument(
        "--outline",
        action="store_true",
        help="also save the photo with the detected corners drawn on it",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.trim < 0.25:
        parser.error("--trim must be in [0, 0.25)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in args.images:
        photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if photo is None:
            print(f"{path}: could not read image", file=sys.stderr)
            failures += 1
            continue

        try:
            result = scan(photo, args.mode, args.trim)
        except DocumentNotFound as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        target = args.out_dir / f"{path.stem}_scan.png"
        cv2.imwrite(str(target), result.image)
        print(f"{path} -> {target}")

        if args.outline:
            outline = args.out_dir / f"{path.stem}_outline.jpg"
            cv2.imwrite(str(outline), draw_outline(photo, result.corners))

    return 1 if failures else 0
