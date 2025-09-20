"""Unit tests for the FastAPI feature detection endpoints."""

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fds import app, encode_overlay, load_image_from_iss, make_detector, params_hash

client = TestClient(app)


def test_load_image_from_iss(mock_iss_response: MagicMock, sample_image: np.ndarray) -> None:
    """Verify images are fetched and decoded from the ISS."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_iss_response.content = encoded.tobytes()

    with patch("requests.get", return_value=mock_iss_response) as mock_get:
        result = load_image_from_iss("test_image_id")

    mock_get.assert_called_once_with("http://localhost:8000/images/test_image_id", timeout=30)
    assert result.shape == sample_image.shape
    assert np.any(result > 0)


def test_make_detector_orb() -> None:
    """Ensure ORB detector creation succeeds."""

    detector = make_detector("orb", {"n_features": 1000, "scaleFactor": 1.1, "nlevels": 4})
    assert isinstance(detector, cv2.ORB)


def test_make_detector_sift() -> None:
    """Ensure SIFT detector creation succeeds."""

    detector = make_detector("sift", {"n_features": 1000})
    assert isinstance(detector, cv2.SIFT)


def test_make_detector_invalid() -> None:
    """Unsupported detector names should raise HTTPException."""

    with pytest.raises(HTTPException) as excinfo:
        make_detector("invalid_algo", {})
    assert "unknown algo" in str(excinfo.value)


def test_params_hash() -> None:
    """Hashes must be stable regardless of dict order."""

    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_encode_overlay(sample_image: np.ndarray) -> None:
    """The overlay encoder should produce a base64 image string."""

    keypoint = cv2.KeyPoint(x=50.0, y=50.0, size=10.0, angle=0, response=1.0, octave=0, class_id=0)
    result = encode_overlay(sample_image, [keypoint])

    assert result.startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.split(",", maxsplit=1)[1])
    assert decoded


def _mock_feature_post(mock_artifact_response: MagicMock) -> MagicMock:
    """Return a configured artifact upload response."""

    mock_artifact_response.json.return_value = {"artifact_id": "test_artifact_id"}
    return mock_artifact_response


@patch("fds.requests.get")
@patch("fds.requests.post")
def test_detect_endpoint(
    mock_post: MagicMock,
    mock_get: MagicMock,
    sample_image: np.ndarray,
    mock_artifact_response: MagicMock,
    mock_iss_response: MagicMock,
) -> None:
    """Integration test for the `/detect` endpoint."""

    _, encoded = cv2.imencode(".png", sample_image)
    mock_iss_response.content = encoded.tobytes()
    mock_get.return_value = mock_iss_response
    mock_post.return_value = _mock_feature_post(mock_artifact_response)

    response = client.post(
        "/detect",
        json={
            "image_id": "test_image",
            "algo": "orb",
            "params": {"n_features": 500},
            "return_overlay": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == "test_image"
    assert data["algo"] == "orb"
    assert "algo_version" in data
    assert "count" in data
    assert "overlay_png" in data

    mock_get.assert_called_once()
    mock_post.assert_called_once()


def test_algos_endpoint() -> None:
    """Verify the `/algos` endpoint advertises supported detectors."""

    response = client.get("/algos")
    assert response.status_code == 200
    data = response.json()
    assert {algo["name"] for algo in data} >= {"orb", "sift"}


def test_health_endpoint() -> None:
    """The health endpoint should return a simple ok payload."""

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
