"""Helper utilities for feature extraction workflows."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CharucoConfig:
    """Configuration parameters for a ChArUco board."""

    squares_x: int = 21
    squares_y: int = 21
    square_length: float = 1.0
    marker_length: float = 0.5
    dictionary: str = "DICT_5X5_1000"
    return_overlay: bool = False


def _odd(value: int) -> int:
    """Return the closest odd integer greater or equal to *value*."""

    value = int(value)
    return value if value % 2 else value + 1


def _resolve_dictionary(dictionary_name: str) -> "cv2.aruco_Dictionary":
    """Resolve the requested ArUco dictionary or raise a descriptive error."""

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Bad dictionary {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _expected_sizes(gray: np.ndarray, config: CharucoConfig) -> Tuple[float, float, float]:
    """Estimate expected marker scales in pixels."""

    height, width = gray.shape[:2]
    perimeter = float(2 * (width + height))
    square_px = min(width / float(config.squares_x), height / float(config.squares_y))
    ratio = config.marker_length / config.square_length if config.square_length else 0.5
    marker_px = max(1.0, square_px * ratio)
    return square_px, marker_px, perimeter


def _build_detector_parameters(
    marker_px: float,
    perimeter: float,
) -> Tuple["cv2.aruco_DetectorParameters", "cv2.aruco_CharucoParameters"]:
    """Create tuned detector parameters for the expected marker size."""

    detector_params = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )

    min_perimeter_rate = max(0.005, 0.6 * (4.0 * marker_px / perimeter))
    win_min = _odd(max(5, int(round(marker_px * 0.6))))
    win_max = _odd(max(win_min + 2, int(round(marker_px * 2.5))))
    win_step = 2 if (win_max - win_min) > 12 else 1
    refine_window = _odd(max(3, int(round(marker_px * 0.15))))

    detector_params.adaptiveThreshWinSizeMin = win_min
    detector_params.adaptiveThreshWinSizeMax = win_max
    detector_params.adaptiveThreshWinSizeStep = win_step
    detector_params.adaptiveThreshConstant = 5
    detector_params.polygonalApproxAccuracyRate = 0.01
    detector_params.minMarkerPerimeterRate = float(min_perimeter_rate)
    detector_params.maxMarkerPerimeterRate = 4.0
    detector_params.minCornerDistanceRate = 0.005
    detector_params.minDistanceToBorder = 1
    detector_params.markerBorderBits = 1
    detector_params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    detector_params.cornerRefinementWinSize = refine_window
    detector_params.cornerRefinementMinAccuracy = 0.05

    if hasattr(detector_params, "detectInvertedMarker"):
        detector_params.detectInvertedMarker = True
    if hasattr(detector_params, "useAruco3Detection"):
        detector_params.useAruco3Detection = True

    charuco_params = (
        cv2.aruco.CharucoParameters() if hasattr(cv2.aruco, "CharucoParameters") else None
    )
    if charuco_params is not None:
        charuco_params.minMarkers = 2
        charuco_params.tryRefineMarkers = True

    return detector_params, charuco_params


def _run_charuco_detection(
    gray: np.ndarray,
    board: "cv2.aruco_CharucoBoard",
    dictionary: "cv2.aruco_Dictionary",
    detector_params: "cv2.aruco_DetectorParameters",
    charuco_params: "cv2.aruco_CharucoParameters | None",
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute one pass of ChArUco detection, returning corners and ids."""

    if hasattr(cv2.aruco, "CharucoDetector"):
        refine_params = cv2.aruco.RefineParameters()
        detector = cv2.aruco.CharucoDetector(
            board,
            charucoParams=charuco_params,
            detectorParams=detector_params,
            refineParams=refine_params,
        )
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or ids is None:
            return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int32)
        return corners, ids

    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int32)

    interpolation = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if interpolation is None or len(interpolation) < 3:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int32)

    return interpolation[1], interpolation[2]


def _detect_with_fallback(
    gray: np.ndarray,
    board: "cv2.aruco_CharucoBoard",
    dictionary: "cv2.aruco_Dictionary",
    detector_params: "cv2.aruco_DetectorParameters",
    charuco_params: "cv2.aruco_CharucoParameters | None",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Charuco detection with post-processing fallbacks."""

    corners, ids = _run_charuco_detection(gray, board, dictionary, detector_params, charuco_params)

    if ids.size == 0:
        detector_params.minMarkerPerimeterRate = max(
            0.003,
            detector_params.minMarkerPerimeterRate * 0.5,
        )
        corners, ids = _run_charuco_detection(
            gray,
            board,
            dictionary,
            detector_params,
            charuco_params,
        )

    if ids.size == 0:
        upscaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        corners, ids = _run_charuco_detection(
            upscaled,
            board,
            dictionary,
            detector_params,
            charuco_params,
        )
        if ids.size:
            corners = corners / 1.5

    return corners, ids


def _refine_corners(gray: np.ndarray, corners: np.ndarray, window: int) -> np.ndarray:
    """Run sub-pixel corner refinement when possible."""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    try:
        cv2.cornerSubPix(gray, corners, (window, window), (-1, -1), criteria)
    except cv2.error:
        return corners
    return corners


def _prepare_charuco_context(
    gray: np.ndarray,
    config: CharucoConfig,
) -> Tuple[
    "cv2.aruco_Dictionary",
    "cv2.aruco_DetectorParameters",
    "cv2.aruco_CharucoParameters | None",
    "cv2.aruco_CharucoBoard",
]:
    """Build reusable Charuco components for detection."""

    dictionary = _resolve_dictionary(config.dictionary)
    _, marker_px, perimeter = _expected_sizes(gray, config)
    detector_params, charuco_params = _build_detector_parameters(marker_px, perimeter)
    board = cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length,
        config.marker_length,
        dictionary,
    )
    return dictionary, detector_params, charuco_params, board


def detect_charuco(
    gray: np.ndarray,
    config: CharucoConfig | None = None,
    **overrides: float | int | str | bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Detect ChArUco corners, returning points, ids, object points and overlay."""

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("detect_charuco expects an uint8 grayscale image")

    if config is None:
        config = CharucoConfig(**overrides)
    elif overrides:
        config = replace(config, **overrides)

    dictionary_obj, detector_params, charuco_params, board = _prepare_charuco_context(gray, config)

    corners, ids = _detect_with_fallback(
        gray,
        board,
        dictionary_obj,
        detector_params,
        charuco_params,
    )

    if ids.size == 0:
        empty_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if config.return_overlay else None
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            empty_overlay,
        )

    refined_corners = _refine_corners(
        gray,
        corners,
        detector_params.cornerRefinementWinSize,
    )

    points = refined_corners.reshape(-1, 2).astype(np.float32)
    ids = ids.reshape(-1).astype(np.int32)
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)[ids]

    overlay = None
    if config.return_overlay:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        try:
            cv2.aruco.drawDetectedCornersCharuco(
                overlay,
                points.reshape(-1, 1, 2),
                ids,
            )
        except cv2.error:
            for point in points:
                cv2.circle(
                    overlay,
                    (int(round(point[0])), int(round(point[1]))),
                    3,
                    (0, 255, 0),
                    -1,
                )

    return points, ids, object_points, overlay
