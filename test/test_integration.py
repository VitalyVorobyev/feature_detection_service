"""Integration tests covering the `/detect` endpoint."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fds import app

client = TestClient(app)


@patch("fds.requests.get")
@patch("fds.requests.post")
def test_end_to_end_feature_detection(
    mock_post: MagicMock,
    mock_get: MagicMock,
    sample_image: np.ndarray,
    mock_artifact_response: MagicMock,
) -> None:
    """Exercise the detect endpoint when features are found."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())
    mock_post.return_value = mock_artifact_response

    response = client.post(
        "/detect",
        json={
            "image_id": "test_image_123",
            "algo": "orb",
            "params": {"n_features": 100, "scaleFactor": 1.1, "nlevels": 4},
            "return_overlay": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["features_artifact_id"] == "test_artifact_id"
    assert data["count"] > 0
    assert data["overlay_png"].startswith("data:image/png;base64,")

    mock_get.assert_called_once_with("http://localhost:8000/images/test_image_123", timeout=30)
    mock_post.assert_called_once()
    assert "file" in mock_post.call_args.kwargs["files"]


@patch("fds.requests.get")
@patch("fds.requests.post")
def test_no_features_detected(mock_post: MagicMock, mock_get: MagicMock) -> None:
    """Ensure pipeline behaves when no descriptors are produced."""

    blank_image = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", blank_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())

    response = client.post(
        "/detect",
        json={"image_id": "blank_image", "algo": "sift", "params": {"n_features": 10}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["summary"] == []
    assert data["features_artifact_id"] is None
    mock_post.assert_not_called()


@pytest.mark.parametrize("algo_name", ["orb", "sift", "akaze", "brisk"])
@patch("fds.requests.get")
@patch("fds.requests.post")
def test_different_algorithms(
    mock_post: MagicMock,
    mock_get: MagicMock,
    algo_name: str,
    sample_image: np.ndarray,
    mock_artifact_response: MagicMock,
) -> None:
    """Check that multiple algorithms run with the same test image."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_get.return_value = MagicMock(status_code=200, content=encoded.tobytes())
    mock_post.return_value = mock_artifact_response

    response = client.post(
        "/detect",
        json={"image_id": f"test_image_{algo_name}", "algo": algo_name, "params": {}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["algo"] == algo_name
    assert data["count"] > 0

    mock_get.reset_mock()
    mock_post.reset_mock()
