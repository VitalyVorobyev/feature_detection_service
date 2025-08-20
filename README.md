# Feature Detection Service (FDS)

[![FDS - Continuous Integration](https://github.com/VitalyVorobyev/feature_detection_service/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/feature_detection_service/actions/workflows/ci.yml)

A (micro)service for detecting and extracting features from images using computer vision algorithms.

## Overview

Feature Detection Service is a FastAPI-based application that provides image feature detection capabilities through a RESTful API. It supports multiple feature detection algorithms from OpenCV such as ORB, SIFT, AKAZE, and BRISK.

The service works in conjunction with an Image Storage Service (ISS) to retrieve images and store feature detection results.

## Features

- Feature detection using multiple algorithms (ORB, SIFT, AKAZE, BRISK)
- Customizable algorithm parameters
- Storage of feature keypoints and descriptors
- Optional visualization with feature overlay generation

## Installation

### Prerequisites

- Python 3.10+
- OpenCV 4.x
- FastAPI
- Uvicorn (for serving the API)

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd fds
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The service requires an Image Storage Service (ISS) to be available. Set the ISS URL through an environment variable:

```bash
# Default: http://localhost:8081
export ISS_URL=http://your-iss-service:port
```

## Running the Service

Start the service using uvicorn:

```bash
uvicorn fds:app --host 0.0.0.0 --port 8080
```

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

### Endpoints

#### POST /detect

Detect features in an image.

**Request Body:**

```json
{
  "image_id": "string",     // ID of the image in the ISS
  "algo": "string",         // Algorithm: "orb", "sift", "akaze", or "brisk"
  "params": {},             // Algorithm-specific parameters
  "return_overlay": false   // Whether to return visualization overlay
}
```

**Response:**

```json
{
  "image_id": "string",
  "algo": "string",
  "algo_version": "string",
  "params_hash": "string",
  "count": 0,               // Number of features detected
  "summary": [],            // Summary of detected features
  "features_artifact_id": "string",
  "overlay_png": "string"   // Base64 encoded PNG (if requested)
}
```

#### GET /algos

Get information about supported algorithms and their parameters.

**Response:**

```json
[
  {
    "name": "orb",
    "params": {
      "n_features": "int",
      "scaleFactor": "float",
      "nlevels": "int"
    }
  },
  {
    "name": "sift",
    "params": {
      "n_features": "int"
    }
  },
  {
    "name": "akaze",
    "params": {}
  },
  {
    "name": "brisk",
    "params": {}
  }
]
```

#### GET /healthz

Health check endpoint.

**Response:**

```json
{
  "ok": true
}
```

## Supported Algorithms

| Algorithm | Parameters | Description |
|-----------|------------|-------------|
| ORB       | n_features, scaleFactor, nlevels | Oriented FAST and Rotated BRIEF |
| SIFT      | n_features | Scale-Invariant Feature Transform |
| AKAZE     | - | Accelerated-KAZE features |
| BRISK     | - | Binary Robust Invariant Scalable Keypoints |

## Algorithm Parameters

### ORB
- `n_features` (int): Maximum number of features to detect (default: 2000)
- `scaleFactor` (float): Pyramid decimation ratio (default: 1.2)
- `nlevels` (int): Number of pyramid levels (default: 8)

### SIFT
- `n_features` (int): Maximum number of features to detect (default: 2000)

### AKAZE and BRISK
No configurable parameters through the API.

## Integration with ISS

The Feature Detection Service depends on an Image Storage Service (ISS) to:
1. Retrieve images by ID
2. Store feature detection results as artifacts

Make sure the ISS is properly configured and accessible at the URL specified in the `ISS_URL` environment variable.

## Example Usage

### Detecting Features

```bash
curl -X 'POST' \
  'http://localhost:8080/detect' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_id": "sample_image_id",
  "algo": "orb",
  "params": {"n_features": 5000},
  "return_overlay": true
}'
```

## License

Under the [MIT License](LICENSE).
