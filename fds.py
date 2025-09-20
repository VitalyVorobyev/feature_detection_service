"""FastAPI application exposing feature and pattern detection endpoints."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import detect_charuco

JsonDict = Dict[str, object]

ISS_URL = os.environ.get("ISS_URL", "http://localhost:8000")
app = FastAPI(title="Feature Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "ETag"],
    max_age=86400,
)


class DetectReq(BaseModel):
    """Request payload for the generic feature detection endpoint."""

    image_id: str
    algo: str = Field(description="orb|sift|akaze|brisk")
    params: Dict[str, object] = Field(default_factory=dict)
    return_overlay: bool = False


class PatternDetectReq(BaseModel):
    """Request payload for calibration pattern detection."""

    image_id: str
    pattern: str = Field(description="charuco|circle_grid|chessboard|apriltag")
    params: Dict[str, object] = Field(default_factory=dict)
    return_overlay: bool = False


def load_image_from_iss(image_id: str) -> np.ndarray:
    """Fetch an image from the ISS and decode it into a BGR array."""

    response = requests.get(f"{ISS_URL}/images/{image_id}", timeout=30)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="image not found")

    buffer = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid image bytes")
    return image


def algo_version(_algo: str) -> str:
    """Return the OpenCV version for the current runtime."""

    return f"opencv-{cv2.__version__}"  # type: ignore[attr-defined]


def make_detector(algo: str, params: Dict[str, object]) -> cv2.Feature2D:
    """Instantiate an OpenCV feature detector using provided parameters."""

    if algo == "orb":
        return cv2.ORB_create(
            nfeatures=int(params.get("n_features", 2000)),
            scaleFactor=float(params.get("scaleFactor", 1.2)),
            nlevels=int(params.get("nlevels", 8)),
        )
    if algo == "sift":
        return cv2.SIFT_create(nfeatures=int(params.get("n_features", 2000)))
    if algo == "akaze":
        return cv2.AKAZE_create()
    if algo == "brisk":
        return cv2.BRISK_create()
    raise HTTPException(status_code=400, detail=f"unknown algo {algo}")


def params_hash(params: Dict[str, object]) -> str:
    """Create a stable hash representing the parameter dictionary."""

    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode(encoding="utf-8")).hexdigest()[:16]


def encode_png(image: np.ndarray) -> str:
    """Encode a numpy image as a base64 data URI."""

    success, png = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="failed to encode overlay")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def encode_overlay(image: np.ndarray, keypoints: Sequence[cv2.KeyPoint]) -> str:
    """Draw detected keypoints on top of the image and encode it."""

    overlay = image.copy()
    cv2.drawKeypoints(  # type: ignore[attr-defined]
        image,
        keypoints,
        overlay,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,  # type: ignore[attr-defined]
    )
    return encode_png(overlay)


def _keypoints_to_array(keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Convert OpenCV keypoints into a compact numpy representation."""

    points = np.zeros((len(keypoints), 7), dtype=np.float32)
    for idx, keypoint in enumerate(keypoints):
        points[idx] = (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            float(keypoint.octave),
            float(getattr(keypoint, "class_id", -1)),
        )
    return points


def _summarise_keypoints(keypoints: Sequence[cv2.KeyPoint], limit: int = 1000) -> List[JsonDict]:
    """Create a UI-friendly keypoint summary limited to the first *limit* items."""

    summary: List[JsonDict] = []
    for keypoint in keypoints[:limit]:
        summary.append(
            {
                "x": float(keypoint.pt[0]),
                "y": float(keypoint.pt[1]),
                "size": float(keypoint.size),
                "angle": float(keypoint.angle),
                "resp": float(keypoint.response),
                "octave": int(keypoint.octave),
            }
        )
    return summary


def _normalise_descriptors(algo: str, descriptors: Optional[np.ndarray]) -> np.ndarray:
    """Ensure descriptors are numpy arrays with consistent dtype."""

    if descriptors is None:
        if algo == "sift":
            return np.zeros((0, 128), dtype=np.float32)
        return np.zeros((0, 32), dtype=np.uint8)

    if algo == "sift":
        return descriptors.astype(np.float32)
    return descriptors.astype(np.uint8)


def _persist_features(
    image_id: str,
    algo: str,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
) -> str:
    """Store keypoints/descriptors bundle in the ISS and return the artifact id."""

    payload = io.BytesIO()
    np.savez_compressed(payload, keypoints=keypoints, descriptors=descriptors)
    files = {"file": ("features.npz", payload.getvalue(), "application/octet-stream")}
    metadata = json.dumps({"image_id": image_id, "algo": algo})
    response = requests.post(
        f"{ISS_URL}/artifacts?kind=features&meta={metadata}",
        files=files,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["artifact_id"]


@app.post("/detect")
def detect(request: DetectReq) -> JsonDict:
    """Detect sparse features and store descriptors for ISS download."""

    image = load_image_from_iss(request.image_id)
    detector = make_detector(request.algo, request.params)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    keypoints = keypoints or []

    normalised_descriptors = _normalise_descriptors(request.algo, descriptors)
    artifact_id = None
    summary: List[JsonDict] = []

    if keypoints:
        kp_array = _keypoints_to_array(keypoints)
        artifact_id = _persist_features(
            request.image_id,
            request.algo,
            kp_array,
            normalised_descriptors,
        )
        summary = _summarise_keypoints(keypoints)

    response: JsonDict = {
        "image_id": request.image_id,
        "algo": request.algo,
        "algo_version": algo_version(request.algo),
        "params_hash": params_hash(request.params),
        "count": len(keypoints),
        "summary": summary,
        "features_artifact_id": artifact_id,
    }
    if request.return_overlay:
        response["overlay_png"] = encode_overlay(image, keypoints)
    return response


def _detect_charuco_points(
    gray: np.ndarray,
    overlay: np.ndarray,
    params: Dict[str, object],
) -> List[JsonDict]:
    """Detect ChArUco corners and return structured point data."""

    corners, ids, obj_points, charuco_overlay = detect_charuco(gray, return_overlay=True, **params)
    if charuco_overlay is not None:
        overlay[:] = charuco_overlay

    points: List[JsonDict] = []
    for idx, corner, obj_pt in zip(ids.flatten(), corners, obj_points):
        points.append(
            {
                "x": float(corner[0]),
                "y": float(corner[1]),
                "id": int(idx),
                "local_x": float(obj_pt[0]),
                "local_y": float(obj_pt[1]),
                "local_z": float(obj_pt[2]),
            }
        )
    return points


def _detect_circle_grid_points(
    gray: np.ndarray,
    overlay: np.ndarray,
    params: Dict[str, object],
) -> List[JsonDict]:
    """Detect symmetric or asymmetric circle grids."""

    rows = int(params.get("rows", 4))
    cols = int(params.get("cols", 5))
    symmetric = bool(params.get("symmetric", True))
    flags = cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID

    found, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
    points: List[JsonDict] = []
    if found:
        cv2.drawChessboardCorners(overlay, (cols, rows), centers, found)
        for idx, center in enumerate(centers):
            col_idx = idx % cols
            row_idx = idx // cols
            points.append(
                {
                    "x": float(center[0][0]),
                    "y": float(center[0][1]),
                    "local_x": float(col_idx),
                    "local_y": float(row_idx),
                }
            )
    return points


def _detect_chessboard_points(
    gray: np.ndarray,
    overlay: np.ndarray,
    params: Dict[str, object],
) -> List[JsonDict]:
    """Detect chessboard corners with subgrid coordinates."""

    rows = int(params.get("rows", 7))
    cols = int(params.get("cols", 7))
    found, corners = cv2.findChessboardCorners(gray, (cols, rows))

    points: List[JsonDict] = []
    if found:
        cv2.drawChessboardCorners(overlay, (cols, rows), corners, found)
        for idx, corner in enumerate(corners):
            col_idx = idx % cols
            row_idx = idx // cols
            points.append(
                {
                    "x": float(corner[0][0]),
                    "y": float(corner[0][1]),
                    "local_x": float(col_idx),
                    "local_y": float(row_idx),
                }
            )
    return points


def _detect_apriltag_points(
    gray: np.ndarray,
    overlay: np.ndarray,
    params: Dict[str, object],
) -> List[JsonDict]:
    """Detect AprilTag markers and return their approximate centres."""

    dictionary_name = params.get("dictionary", "DICT_APRILTAG_36h11")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(dictionary_name)))
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    points: List[JsonDict] = []
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        for corner_group, marker_id in zip(corners, ids.flatten()):
            centre = corner_group[0].mean(axis=0)
            points.append(
                {
                    "x": float(centre[0]),
                    "y": float(centre[1]),
                    "id": int(marker_id),
                    "local_x": 0.0,
                    "local_y": 0.0,
                }
            )
    return points


PatternDetector = Callable[[np.ndarray, np.ndarray, Dict[str, object]], List[JsonDict]]
PATTERN_DETECTORS: Dict[str, PatternDetector] = {
    "charuco": _detect_charuco_points,
    "circle_grid": _detect_circle_grid_points,
    "chessboard": _detect_chessboard_points,
    "apriltag": _detect_apriltag_points,
}


@app.post("/detect_pattern")
def detect_pattern(request: PatternDetectReq) -> JsonDict:
    """Detect calibration pattern points and optionally an overlay."""

    if request.pattern not in PATTERN_DETECTORS:
        raise HTTPException(status_code=400, detail=f"unknown pattern {request.pattern}")

    image = load_image_from_iss(request.image_id)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    overlay = image.copy()

    detector = PATTERN_DETECTORS[request.pattern]
    points = detector(gray, overlay, request.params)

    response: JsonDict = {
        "image_id": request.image_id,
        "pattern": request.pattern,
        "algo_version": algo_version(request.pattern),
        "params_hash": params_hash(request.params),
        "count": len(points),
        "points": points,
    }
    if request.return_overlay:
        response["overlay_png"] = encode_png(overlay)
    return response


@app.get("/algos")
def algos() -> List[JsonDict]:
    """Return the supported sparse feature algorithms and their parameters."""

    return [
        {
            "name": "orb",
            "params": {"n_features": "int", "scaleFactor": "float", "nlevels": "int"},
        },
        {"name": "sift", "params": {"n_features": "int"}},
        {"name": "akaze", "params": {}},
        {"name": "brisk", "params": {}},
    ]


@app.get("/patterns")
def patterns() -> List[JsonDict]:
    """Return the supported calibration pattern detectors."""

    return [
        {
            "name": "charuco",
            "params": {
                "squares_x": "int",
                "squares_y": "int",
                "square_length": "float",
                "marker_length": "float",
                "dictionary": "str",
            },
        },
        {
            "name": "circle_grid",
            "params": {"rows": "int", "cols": "int", "symmetric": "bool"},
        },
        {"name": "chessboard", "params": {"rows": "int", "cols": "int"}},
        {"name": "apriltag", "params": {"dictionary": "str"}},
    ]


@app.get("/healthz")
def health() -> JsonDict:
    """Return application health diagnostics."""

    return {"ok": True}
