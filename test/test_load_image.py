"""Tests covering image loading utilities."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from fds import load_image_from_iss

class TestLoadImageFromISS:
    """Group of unit tests for :func:`fds.load_image_from_iss`."""

    def test_successful_image_load(self, sample_image, mock_iss_response):
        """Test successfully loading an image from ISS service"""
        # Encode the sample image to simulate the image response
        _, img_encoded = cv2.imencode(".png", sample_image)
        mock_iss_response.content = img_encoded.tobytes()

        with patch("requests.get", return_value=mock_iss_response) as mock_get:
            # Call the function with a dummy image ID
            result = load_image_from_iss("test_image_id")

            # Assert the mock was called with expected URL
            mock_get.assert_called_once_with(
                "http://localhost:8000/images/test_image_id",
                timeout=30,
            )

            # Check that returned image has the same shape and content as our sample
            assert result.shape == sample_image.shape
            assert np.array_equal(result, sample_image)

    def test_image_not_found(self, mock_iss_response):
        """Test handling when image is not found"""
        # Set up mock response for 404
        mock_iss_response.status_code = 404

        with patch("requests.get", return_value=mock_iss_response):
            # Should raise HTTPException with 404 status
            with pytest.raises(HTTPException) as excinfo:
                load_image_from_iss("non_existent_image")

            assert excinfo.value.status_code == 404
            assert "image not found" in str(excinfo.value.detail)

    def test_invalid_image_bytes(self, mock_iss_response):
        """Test handling when image bytes are invalid"""
        # Set up mock response with invalid image bytes
        mock_iss_response.status_code = 200
        mock_iss_response.content = b"not an image"

        with patch("requests.get", return_value=mock_iss_response):
            # Should raise HTTPException with 400 status
            with pytest.raises(HTTPException) as excinfo:
                load_image_from_iss("invalid_image")

            assert excinfo.value.status_code == 400
            assert "invalid image bytes" in str(excinfo.value.detail)

    def test_custom_iss_url(self, sample_image, mock_iss_response):
        """Test using a custom ISS URL from environment variable"""
        # Encode the sample image to simulate the image response
        _, img_encoded = cv2.imencode(".png", sample_image)
        mock_iss_response.content = img_encoded.tobytes()

        with patch("os.environ.get", return_value="https://custom-iss.example.com"):
            with patch("requests.get", return_value=mock_iss_response) as mock_get:
                # Reset the ISS_URL in the module
                with patch("fds.ISS_URL", "https://custom-iss.example.com"):
                    result = load_image_from_iss("test_image_id")

                # Assert the mock was called with the custom URL
                mock_get.assert_called_once_with(
                    "https://custom-iss.example.com/images/test_image_id",
                    timeout=30
                )

                # Check that returned image has the same shape as our sample
                assert result.shape == sample_image.shape
                assert np.any(result > 0)  # Should have some non-zero pixels
