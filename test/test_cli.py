import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import fds_cli


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cli_orb_detection(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # random texture gives ORB plenty of keypoints
    img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    image_path = image_dir / "noise.png"
    assert cv2.imwrite(str(image_path), img)

    output_file = "result.json"
    config = {
        "image_directory": "images",
        "feature_type": "orb",
        "feature_params": {"n_features": 300},
        "output_file": output_file,
    }

    config_path = _write_config(tmp_path / "config.json", config)
    fds_cli.main([str(config_path)])

    result_path = (config_path.parent / output_file).resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))

    assert data["feature_type"] == "orb"
    assert data["algo_version"].startswith("opencv-")
    assert len(data["images"]) == 1
    first = data["images"][0]
    assert first["count"] > 0
    assert len(first["descriptors"]) == first["count"]
    assert first["file"].endswith("noise.png")


def test_cli_charuco_detection(tmp_path, monkeypatch):
    image_dir = tmp_path / "charuco_imgs"
    image_dir.mkdir()

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    image_path = image_dir / "blank.png"
    assert cv2.imwrite(str(image_path), img)

    def fake_detect(gray, **kwargs):
        points = np.array([[10.0, 15.0]], dtype=np.float32)
        ids = np.array([42], dtype=np.int32)
        objs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        return points, ids, objs, None

    monkeypatch.setattr("fds_cli.detect_charuco", fake_detect)

    config = {
        "image_directory": "charuco_imgs",
        "feature_type": "charuco",
        "feature_params": {
            "squares_x": 5,
            "squares_y": 7,
            "square_length": 1.0,
            "marker_length": 0.5,
            "dictionary": "DICT_5X5_1000",
        },
        "output_file": "charuco.json",
    }

    config_path = _write_config(tmp_path / "charuco_config.json", config)
    fds_cli.main([str(config_path)])

    result_path = config_path.parent / "charuco.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))

    assert data["feature_type"] == "charuco"
    assert data["params_hash"]
    assert len(data["images"]) == 1
    points = data["images"][0]["points"]
    assert len(points) == 1
    assert points[0]["id"] == 42
    assert points[0]["local_z"] == 0.0


@pytest.mark.parametrize("bad_field", ["image_directory", "feature_type", "feature_params", "output_file"])
def test_cli_missing_config_field(tmp_path, bad_field):
    config = {
        "image_directory": "images",
        "feature_type": "orb",
        "feature_params": {},
        "output_file": "out.json",
    }
    config.pop(bad_field)
    config_path = _write_config(tmp_path / "missing.json", config)

    with pytest.raises(ValueError):
        fds_cli.run_from_config(config_path)
