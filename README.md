# Feature Detection Service (FDS)

[![FDS - Continuous Integration](https://github.com/VitalyVorobyev/feature_detection_service/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/feature_detection_service/actions/workflows/ci.yml) [![Pylint](https://img.shields.io/badge/pylint-checked-brightgreen)](https://github.com/PyCQA/pylint) [![codecov](https://codecov.io/gh/VitalyVorobyev/feature_detection_service/branch/main/graph/badge.svg)](https://codecov.io/gh/VitalyVorobyev/feature_detection_service)

A toolkit for detecting and extracting features from images using computer vision algorithms.

## Overview

Feature Detection Service is a FastAPI-based application and companion CLI that provide image feature and pattern detection capabilities. It supports multiple feature detection algorithms from OpenCV such as ORB, SIFT, AKAZE, and BRISK as well as pattern detectors for ChArUco boards, circle grids, chessboards, and AprilTags.

The service works in conjunction with an Image Storage Service (ISS) to retrieve images and store feature detection results.

## Features

- Feature detection using multiple algorithms (ORB, SIFT, AKAZE, BRISK)
- Pattern detection for calibration targets (ChArUco, circle grid, chessboard, AprilTag)
- Config-driven CLI for batch feature extraction with progress feedback (tqdm) and rich logging (loguru/stdlib)
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
   git clone https://github.com/VitalyVorobyev/feature_detection_service.git
   cd feature_detection_service
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The service requires an [Image Storage Service (ISS)](https://github.com/VitalyVorobyev/image_store_service) to be available. Set the ISS URL through an environment variable:

```bash
# Default: http://localhost:8000
export ISS_URL=http://your-iss-service:port
```

Container deployments can reuse the provided `example.env` file. Update the `ISS_URL` value before loading it so the service can reach your Image Storage Service. The published Docker image also accepts an `ISS_URL` build argument, defaulting to `http://localhost:8000` when not supplied.

## Command Line Interface

Alongside the API, the repository provides a JSON-driven CLI in `fds_cli.py` for offline or batch processing. The CLI uses `tqdm` to show progress bars and prefers `loguru` for structured console output (falling back to the Python logging module automatically).

### Usage

```bash
python fds_cli.py path/to/config.json --log-level INFO
```

Use `--log-level DEBUG` to surface per-image details that complement the progress bar.

### Configuration Schema

Each config file is a JSON object with the following keys:

- `image_directory` (str): Folder containing images to process.
- `feature_type` (str): One of `orb`, `sift`, `akaze`, `brisk`, or `charuco`.
- `feature_params` (object): Parameters forwarded to the detector (pass `{}` when none are required).
- `output_file` (str): Path (relative to the config file) for the generated JSON results.
- `extensions` (array of str, optional): Restrict the image extensions to scan (defaults to common formats).

### Example Configs

Ready-to-tweak examples live in `examples/cli_configs/`:

- `examples/cli_configs/orb_features.json` – ORB keypoints with elevated feature count.
- `examples/cli_configs/sift_features.json` – SIFT descriptors for TIFF/PNG datasets.
- `examples/cli_configs/akaze_features.json` – AKAZE keypoints with default parameters.
- `examples/cli_configs/charuco_board.json` – ChArUco board corner extraction with board geometry specified.

Adjust the `image_directory` to point at your data and select an `output_file` location before running the CLI.

### Linting

Run pylint locally with the helper script:

```bash
scripts/run_pylint.sh
```

The CI workflow executes the same script, so keeping it clean locally avoids surprises in pull requests.

If you plan to use the CLI with the optional `loguru` dependency, ensure your Python environment has development headers available before installing requirements. For example, on Debian/Ubuntu-based systems:

```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

Recreate (or reinstall inside) your virtual environment afterwards:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

#### POST /detect_pattern

Detect calibration patterns such as ChArUco boards, circle grids, chessboards and AprilTags.

**Request Body:**

```json
{
  "image_id": "string",      // ID of the image in the ISS
  "pattern": "string",       // "charuco", "circle_grid", "chessboard", or "apriltag"
  "params": {},              // Pattern-specific parameters
  "return_overlay": false    // Whether to return visualization overlay
}
```

**Response:**

```json
{
  "image_id": "string",
  "pattern": "string",
  "algo_version": "string",
  "params_hash": "string",
  "count": 0,               // Number of pattern points detected
  "points": [               // Detected points with optional IDs
    {
      "x": 0.0,
      "y": 0.0,
      "local_x": 0.0,
      "local_y": 0.0,
      "local_z": 0.0,      // local_z may be omitted for planar patterns
      "id": 0              // present for Charuco and AprilTag
    }
  ],
  "overlay_png": "string"   // Base64 encoded PNG (if requested)
}
```

#### GET /patterns

Get information about supported pattern detectors and their parameters.

**Response:**

```json
[
  {
    "name": "charuco",
    "params": {
      "squares_x": "int",
      "squares_y": "int",
      "square_length": "float",
      "marker_length": "float",
      "dictionary": "str"
    }
  },
  {
    "name": "circle_grid",
    "params": {
      "rows": "int",
      "cols": "int",
      "symmetric": "bool"
    }
  },
  {
    "name": "chessboard",
    "params": {
      "rows": "int",
      "cols": "int"
    }
  },
  {
    "name": "apriltag",
    "params": {
      "dictionary": "str"
    }
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

## Supported Pattern Detectors

| Pattern | Parameters | Description |
|---------|------------|-------------|
| charuco | squares_x, squares_y, square_length, marker_length, dictionary | ChArUco board corner detection |
| circle_grid | rows, cols, symmetric | Circle grid center detection |
| chessboard | rows, cols | Chessboard corner detection |
| apriltag | dictionary | AprilTag marker detection |

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

### Detecting Patterns

```bash
curl -X 'POST' \
  'http://localhost:8080/detect_pattern' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_id": "sample_image_id",
  "pattern": "chessboard",
  "params": {"rows": 7, "cols": 7},
  "return_overlay": true
}'
```

### CLI (Batch Processing)

```bash
python fds_cli.py examples/cli_configs/orb_features.json --log-level INFO
```

The command reads the example config, walks the `image_directory`, and writes detections to the configured output path while displaying a progress bar when multiple images are present.

## License

Under the [MIT License](LICENSE).
