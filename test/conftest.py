import pytest
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_image():
    """Create a simple test image with features that can be detected"""
    # Creating a 100x100 black image with some white rectangles
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:40, 20:40] = 255  # White square
    img[60:80, 60:80] = 255  # Another white square
    return img

@pytest.fixture
def mock_iss_response():
    """Create a mock response object for ISS service"""
    mock = MagicMock()
    mock.status_code = 200
    return mock

@pytest.fixture
def mock_artifact_response():
    """Create a mock response for artifact creation"""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"artifact_id": "test_artifact_id"}
    return mock
