import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, ANY
import json
import io
from fastapi.testclient import TestClient

from fds import app, load_image_from_iss, DetectReq

class TestIntegration:
    """Integration tests that mock external service dependencies"""
    
    @patch('fds.requests.get')
    @patch('fds.requests.post')
    def test_end_to_end_feature_detection(self, mock_post, mock_get, sample_image, mock_artifact_response):
        """
        Test the complete feature detection workflow by mocking external service calls
        """
        # Create a test client
        client = TestClient(app)
        
        # Setup mock for image loading
        _, img_encoded = cv2.imencode('.png', sample_image)
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = img_encoded.tobytes()
        mock_get.return_value = mock_get_response
        
        # Setup mock for artifact storage
        mock_post.return_value = mock_artifact_response
        
        # Test request for ORB algorithm with custom parameters
        response = client.post(
            "/detect",
            json={
                "image_id": "test_image_123",
                "algo": "orb", 
                "params": {"n_features": 100, "scaleFactor": 1.1, "nlevels": 4},
                "return_overlay": True
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == "test_image_123"
        assert data["algo"] == "orb"
        assert data["count"] > 0  # Should detect some features
        assert "overlay_png" in data
        assert data["overlay_png"].startswith("data:image/png;base64,")
        assert data["features_artifact_id"] == "test_artifact_id"
        
        # Verify mock calls
        mock_get.assert_called_once_with("http://localhost:8081/images/test_image_123", timeout=30)
        mock_post.assert_called_once()
        # Verify the POST contains the expected metadata
        assert "meta=" in mock_post.call_args[0][0]
        # The file content is binary, so we should just check the POST was called with files
        assert "file" in mock_post.call_args[1]["files"]
    
    @patch('fds.requests.get')
    @patch('fds.requests.post')
    def test_no_features_detected(self, mock_post, mock_get):
        """
        Test handling when no features are detected in the image
        """
        # Create a test client
        client = TestClient(app)
        
        # Create a blank image with no features
        blank_image = np.zeros((50, 50, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode('.png', blank_image)
        
        # Setup mock for image loading
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = img_encoded.tobytes()
        mock_get.return_value = mock_get_response
        
        # Test request
        response = client.post(
            "/detect",
            json={
                "image_id": "blank_image",
                "algo": "sift",
                "params": {"n_features": 10}
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["summary"] == []
        assert data["features_artifact_id"] is None
        
        # Verify that POST was not called (no artifacts to store)
        mock_post.assert_not_called()
    
    @pytest.mark.parametrize("algo_name", ["orb", "sift", "akaze", "brisk"])
    @patch('fds.requests.get')
    @patch('fds.requests.post')
    def test_different_algorithms(self, mock_post, mock_get, algo_name, sample_image, mock_artifact_response):
        """
        Test each supported algorithm with the same sample image
        """
        # Create a test client
        client = TestClient(app)
        
        # Setup mock for image loading
        _, img_encoded = cv2.imencode('.png', sample_image)
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = img_encoded.tobytes()
        mock_get.return_value = mock_get_response
        
        # Setup mock for artifact storage
        mock_post.return_value = mock_artifact_response
        
        # Test request with the current algorithm
        response = client.post(
            "/detect",
            json={
                "image_id": f"test_image_{algo_name}",
                "algo": algo_name,
                "params": {}
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["algo"] == algo_name
        assert "algo_version" in data
        assert data["count"] > 0  # All algorithms should detect features in our sample image
        
        # Reset mocks for the next iteration
        mock_get.reset_mock()
        mock_post.reset_mock()
