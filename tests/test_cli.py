import cv2
import numpy as np
import pytest

from docscan.cli import main
from tests.synthetic import make_scene


def test_cli_writes_scan_and_outline(tmp_path):
    photo, _ = make_scene()
    src = tmp_path / "receipt.jpg"
    cv2.imwrite(str(src), photo)
    out_dir = tmp_path / "out"

    code = main([str(src), "-m", "bw", "-o", str(out_dir), "--outline"])

    assert code == 0
    scanned = cv2.imread(str(out_dir / "receipt_scan.png"), cv2.IMREAD_GRAYSCALE)
    assert scanned is not None
    assert (out_dir / "receipt_outline.jpg").exists()


def test_cli_reports_unreadable_file(tmp_path, capsys):
    code = main([str(tmp_path / "missing.jpg"), "-o", str(tmp_path / "out")])
    assert code == 1
    assert "could not read" in capsys.readouterr().err


def test_cli_reports_missing_document(tmp_path, capsys):
    src = tmp_path / "wall.png"
    cv2.imwrite(str(src), np.full((400, 500, 3), 90, np.uint8))
    code = main([str(src), "-o", str(tmp_path / "out")])
    assert code == 1
    assert "no four-sided" in capsys.readouterr().err


def test_cli_continues_after_a_failure(tmp_path):
    good = tmp_path / "good.jpg"
    cv2.imwrite(str(good), make_scene()[0])
    code = main([str(tmp_path / "nope.jpg"), str(good), "-o", str(tmp_path / "out")])
    assert code == 1
    assert (tmp_path / "out" / "good_scan.png").exists()


def test_cli_rejects_out_of_range_trim(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.jpg"), "--trim", "0.5"])
