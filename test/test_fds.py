import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import json
import cv2
import base64
import io
from fastapi.testclient import TestClient

from fds import app, load_image_from_iss, make_detector, params_hash, encode_overlay, DetectReq

client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a simple test image"""
    # Creating a 100x100 black image with some white rectangles
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:40, 20:40] = 255  # White square
    img[60:80, 60:80] = 255  # Another white square
    return img

@pytest.fixture
def mock_response():
    """Create a mock response object"""
    mock = MagicMock()
    mock.status_code = 200
    return mock

def test_load_image_from_iss(mock_response, sample_image):
    """Test the load_image_from_iss function with mocked requests"""
    # Encode the sample image to simulate the image response
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_response.content = img_encoded.tobytes()
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        # Call the function with a dummy image ID
        result = load_image_from_iss('test_image_id')
        
        # Assert the mock was called with expected URL
        mock_get.assert_called_once_with(f"http://localhost:8081/images/test_image_id", timeout=30)
        
        # Check that returned image has the same shape as our sample
        assert result.shape == sample_image.shape
        # Basic validation that image processing worked
        assert np.any(result > 0)  # Should have some non-zero pixels

def test_make_detector_orb():
    """Test creating an ORB detector"""
    detector = make_detector("orb", {"n_features": 1000, "scaleFactor": 1.1, "nlevels": 4})
    assert detector is not None
    assert isinstance(detector, cv2.ORB)

def test_make_detector_sift():
    """Test creating a SIFT detector"""
    detector = make_detector("sift", {"n_features": 1000})
    assert detector is not None
    assert isinstance(detector, cv2.SIFT)

def test_make_detector_invalid():
    """Test creating an invalid detector raises an exception"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        make_detector("invalid_algo", {})
    assert "unknown algo" in str(excinfo.value)

def test_params_hash():
    """Test the params_hash function"""
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})  # Order doesn't matter
    assert params_hash({"a": 1}) != params_hash({"a": 2})  # Different values yield different hashes

def test_encode_overlay(sample_image):
    """Test the encode_overlay function"""
    # Create a simple keypoint
    kp = cv2.KeyPoint(x=50.0, y=50.0, size=10.0, angle=0, response=1.0, octave=0, class_id=0)
    result = encode_overlay(sample_image, [kp])
    
    # Should return a base64 string with data URI prefix
    assert result.startswith("data:image/png;base64,")
    # Decode the base64 part to ensure it's valid
    base64_str = result.split(",")[1]
    decoded = base64.b64decode(base64_str)
    assert len(decoded) > 0

@patch('fds.requests.get')
@patch('fds.requests.post')
def test_detect_endpoint(mock_post, mock_get, sample_image, mock_response):
    """Test the /detect endpoint with mocked dependencies"""
    # Setup mock for image loading
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_response.content = img_encoded.tobytes()
    mock_get.return_value = mock_response
    
    # Setup mock for posting to artifacts
    post_response = MagicMock()
    post_response.status_code = 200
    post_response.json.return_value = {"artifact_id": "test_artifact_id"}
    mock_post.return_value = post_response
    
    # Test request
    response = client.post(
        "/detect",
        json={"image_id": "test_image", "algo": "orb", "params": {"n_features": 500}, "return_overlay": True}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == "test_image"
    assert data["algo"] == "orb"
    assert "algo_version" in data
    assert "count" in data
    assert "overlay_png" in data
    
    # Verify the mocks were called correctly
    mock_get.assert_called_once()
    assert "test_image" in mock_get.call_args[0][0]
    mock_post.assert_called_once()
    assert "artifacts" in mock_post.call_args[0][0]

def test_algos_endpoint():
    """Test the /algos endpoint"""
    response = client.get("/algos")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 4  # Should have at least 4 algorithms
    assert "orb" in [algo["name"] for algo in data]
    assert "sift" in [algo["name"] for algo in data]

def test_health_endpoint():
    """Test the /healthz endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
