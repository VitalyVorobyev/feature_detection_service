import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from fds import app

client = TestClient(app)

@patch('fds.detect_charuco')
@patch('fds.requests.get')
def test_detect_pattern_charuco(mock_get, mock_detect_charuco, sample_image):
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.content = img_encoded.tobytes()
    mock_get.return_value = mock_resp
    # mock_detect_charuco returns (corners, ids, objpts, overlay)
    mock_detect_charuco.return_value = (
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.array([1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        None
    )
    response = client.post('/detect_pattern', json={'image_id':'id1','pattern':'charuco','params':{}})
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 1
    assert data['points'][0]['id'] == 1
    assert data['points'][0]['local_x'] == 0.0
    assert data['points'][0]['local_y'] == 0.0

@patch('fds.cv2.findCirclesGrid')
@patch('fds.requests.get')
def test_detect_pattern_circle_grid(mock_get, mock_find, sample_image):
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.content = img_encoded.tobytes()
    mock_get.return_value = mock_resp
    mock_find.return_value = (True, np.array([[[1.0,2.0]]], dtype=np.float32))
    response = client.post('/detect_pattern', json={'image_id':'id2','pattern':'circle_grid','params':{'rows':1,'cols':1}})
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 1
    assert data['points'][0]['local_x'] == 0.0
    assert data['points'][0]['local_y'] == 0.0

@patch('fds.cv2.findChessboardCorners')
@patch('fds.requests.get')
def test_detect_pattern_chessboard(mock_get, mock_find, sample_image):
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.content = img_encoded.tobytes()
    mock_get.return_value = mock_resp
    mock_find.return_value = (True, np.array([[[1.0,2.0]]], dtype=np.float32))
    response = client.post('/detect_pattern', json={'image_id':'id3','pattern':'chessboard','params':{'rows':1,'cols':1}})
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 1
    assert data['points'][0]['local_x'] == 0.0
    assert data['points'][0]['local_y'] == 0.0

@patch('fds.cv2.aruco.detectMarkers')
@patch('fds.requests.get')
def test_detect_pattern_apriltag(mock_get, mock_detect, sample_image):
    _, img_encoded = cv2.imencode('.png', sample_image)
    mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.content = img_encoded.tobytes()
    mock_get.return_value = mock_resp
    mock_detect.return_value = ([np.zeros((1,4,2), dtype=np.float32)], np.array([[3]]), None)
    response = client.post('/detect_pattern', json={'image_id':'id4','pattern':'apriltag','params':{}})
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 1
    assert data['points'][0]['id'] == 3
    assert data['points'][0]['local_x'] == 0.0
    assert data['points'][0]['local_y'] == 0.0
