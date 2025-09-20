"""Tests for the pattern detection endpoint."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from fds import app

client = TestClient(app)


@patch("fds.detect_charuco")
@patch("fds.requests.get")
def test_detect_pattern_charuco(
    mock_get: MagicMock,
    mock_detect_charuco: MagicMock,
    sample_image: np.ndarray,
) -> None:
    """The endpoint should forward Charuco detections with metadata."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_resp = MagicMock(status_code=200, content=encoded.tobytes())
    mock_get.return_value = mock_resp
    mock_detect_charuco.return_value = (
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.array([1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        None,
    )

    response = client.post(
        "/detect_pattern",
        json={"image_id": "id1", "pattern": "charuco", "params": {}},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert data["points"][0]["id"] == 1
    assert data["points"][0]["local_x"] == 0.0


@patch("fds.cv2.findCirclesGrid")
@patch("fds.requests.get")
def test_detect_pattern_circle_grid(
    mock_get: MagicMock,
    mock_find: MagicMock,
    sample_image: np.ndarray,
) -> None:
    """Circle grid detection should return lattice coordinates."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())
    mock_find.return_value = (True, np.array([[[1.0, 2.0]]], dtype=np.float32))

    response = client.post(
        "/detect_pattern",
        json={"image_id": "id2", "pattern": "circle_grid", "params": {"rows": 1, "cols": 1}},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert data["points"][0]["local_x"] == 0.0
    assert data["points"][0]["local_y"] == 0.0


@patch("fds.cv2.findChessboardCorners")
@patch("fds.requests.get")
def test_detect_pattern_chessboard(
    mock_get: MagicMock,
    mock_find: MagicMock,
    sample_image: np.ndarray,
) -> None:
    """Chessboard detection should capture points in order."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())
    mock_find.return_value = (True, np.array([[[1.0, 2.0]]], dtype=np.float32))

    response = client.post(
        "/detect_pattern",
        json={"image_id": "id3", "pattern": "chessboard", "params": {"rows": 1, "cols": 1}},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert data["points"][0]["local_x"] == 0.0
    assert data["points"][0]["local_y"] == 0.0


@patch("fds.cv2.aruco.detectMarkers")
@patch("fds.requests.get")
def test_detect_pattern_apriltag(
    mock_get: MagicMock,
    mock_detect: MagicMock,
    sample_image: np.ndarray,
) -> None:
    """AprilTag detection should return approximate marker centres."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())
    mock_detect.return_value = ([np.zeros((1, 4, 2), dtype=np.float32)], np.array([[3]]), None)

    response = client.post(
        "/detect_pattern",
        json={"image_id": "id4", "pattern": "apriltag", "params": {}},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert data["points"][0]["id"] == 3
    assert data["points"][0]["local_x"] == 0.0
