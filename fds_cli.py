"""Command line interface for batch feature detection."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np

from fds import algo_version, algos, make_detector, params_hash
from features import detect_charuco

try:  # pragma: no cover - optional dependency
    import spdlog  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spdlog = None

SPDLOG_LOGGER = None
if spdlog is not None:  # pragma: no cover - optional dependency
    try:
        SPDLOG_LOGGER = spdlog.ConsoleLogger("fds-cli")
    except RuntimeError:
        SPDLOG_LOGGER = spdlog.get("fds-cli")

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm  # type: ignore

    def iter_with_progress(items: Iterable[Path]) -> Iterable[Path]:
        """Wrap an iterable with a tqdm progress bar when available."""

        items = list(items)
        if len(items) <= 1:
            return items
        return tqdm(items, desc="Processing images", unit="image")  # type: ignore[return-value]

except ImportError:  # pragma: no cover - optional dependency

    def iter_with_progress(items: Iterable[Path]) -> Iterable[Path]:
        """Fallback that simply returns the iterable when tqdm is missing."""

        return items


class LoggerProxy:
    """Adapter that hides logging backend differences."""

    def __init__(self) -> None:
        self._backend = logging.getLogger("fds_cli")

    def configure(self, level_name: str) -> None:
        """Update the underlying backend depending on optional spdlog support."""

        level = level_name.upper()
        if SPDLOG_LOGGER is not None:
            log_levels = getattr(spdlog, "LogLevel")
            log_level = getattr(log_levels, level, getattr(log_levels, "INFO"))
            SPDLOG_LOGGER.set_level(log_level)
            self._backend = SPDLOG_LOGGER
            return

        python_level = getattr(logging, level, logging.INFO)
        logging.basicConfig(
            level=python_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        backend = logging.getLogger("fds_cli")
        backend.setLevel(python_level)
        self._backend = backend

    def info(self, message: str, *args: Any) -> None:
        """Log an informational message."""

        self._emit("info", message, *args)

    def debug(self, message: str, *args: Any) -> None:
        """Log a debug message."""

        self._emit("debug", message, *args)

    def warning(self, message: str, *args: Any) -> None:
        """Log a warning message."""

        self._emit("warning", message, *args)

    def error(self, message: str, *args: Any) -> None:
        """Log an error message."""

        self._emit("error", message, *args)

    def _emit(self, method: str, message: str, *args: Any) -> None:
        """Dispatch a log message via the configured backend."""

        if args:
            message = message % args
        logger_fn = getattr(self._backend, method, None)
        if logger_fn is not None:
            logger_fn(message)


LOGGER = LoggerProxy()


@dataclass
class CliConfig:
    """Resolved CLI configuration values."""

    image_directory: Path
    output_file: Path
    feature_type: str
    feature_params: Dict[str, Any]
    extensions: Optional[List[str]]


def _load_config(path: Path) -> Dict[str, Any]:
    """Load and validate the JSON configuration file."""

    with path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")

    required = {"image_directory", "feature_type", "feature_params", "output_file"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Config missing required fields: {', '.join(sorted(missing))}")
    return data


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    """Return an absolute path derived from *maybe_relative*."""

    candidate = Path(maybe_relative)
    if not candidate.is_absolute():
        return (base / candidate).resolve()
    return candidate


def _collect_images(directory: Path, extensions: Optional[List[str]]) -> List[Path]:
    """Gather candidate images from *directory* filtering by extension."""

    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {directory}")

    valid_exts = extensions or [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    normalised = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (ext_name.lower() for ext_name in valid_exts)
    }

    images = [
        path
        for path in sorted(directory.glob("**/*"))
        if path.is_file() and path.suffix.lower() in normalised
    ]
    if not images:
        extension_list = ", ".join(sorted(normalised))
        raise FileNotFoundError(
            f"No images found in {directory} with extensions {extension_list}"
        )
    return images


def _serialize_keypoints(keypoints: Iterable[cv2.KeyPoint]) -> List[Dict[str, Any]]:
    """Convert OpenCV keypoints into JSON-serialisable structures."""

    serialised: List[Dict[str, Any]] = []
    for keypoint in keypoints:
        serialised.append(
            {
                "x": float(keypoint.pt[0]),
                "y": float(keypoint.pt[1]),
                "size": float(keypoint.size),
                "angle": float(keypoint.angle),
                "response": float(keypoint.response),
                "octave": int(keypoint.octave),
                "class_id": int(getattr(keypoint, "class_id", -1)),
            }
        )
    return serialised


def _relative_path(root: Path, file_path: Path) -> str:
    """Return a path relative to *root* when possible."""

    try:
        return str(file_path.relative_to(root))
    except ValueError:
        return str(file_path)


def _detect_keypoints(image: np.ndarray, algo_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a keypoint detector and return serialised results."""

    detector = make_detector(algo_name, params)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    keypoints = keypoints or []

    serialized_descriptors: List[List[float]]
    if descriptors is None:
        serialized_descriptors = []
    else:
        dtype = np.float32 if algo_name == "sift" else np.uint8
        serialized_descriptors = descriptors.astype(dtype).tolist()

    return {
        "count": len(keypoints),
        "keypoints": _serialize_keypoints(keypoints),
        "descriptors": serialized_descriptors,
    }


def _detect_charuco(image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run ChArUco detection and return point descriptors."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points, ids, obj_points, _ = detect_charuco(gray, return_overlay=False, **params)

    detected: List[Dict[str, Any]] = []
    for idx, point, obj in zip(ids.flatten(), points, obj_points):
        detected.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "id": int(idx),
                "local_x": float(obj[0]),
                "local_y": float(obj[1]),
                "local_z": float(obj[2]),
            }
        )
    return {"count": len(detected), "points": detected}


def _is_supported_algo(name: str) -> bool:
    """Check whether *name* maps to a supported detector."""

    supported = {entry["name"] for entry in algos()}
    supported.add("charuco")
    return name in supported


def _resolve_config(config_path: Path) -> CliConfig:
    """Resolve the raw JSON config into a structured object."""

    LOGGER.info("Loading configuration from %s", config_path)
    raw = _load_config(config_path)
    base_dir = config_path.parent

    image_directory = _resolve_path(base_dir, str(raw["image_directory"]))
    output_file = _resolve_path(base_dir, str(raw["output_file"]))
    feature_type = str(raw["feature_type"]).lower()
    feature_params = dict(raw.get("feature_params", {}))
    extensions = raw.get("extensions")
    if extensions is not None:
        extensions = [str(ext) for ext in extensions]

    if not _is_supported_algo(feature_type):
        raise ValueError(f"Unsupported feature type '{feature_type}'")

    LOGGER.info("Resolved image directory: %s", image_directory)
    LOGGER.info("Resolved output file: %s", output_file)
    if feature_params:
        LOGGER.debug("Feature parameters: %s", feature_params)

    return CliConfig(
        image_directory=image_directory,
        output_file=output_file,
        feature_type=feature_type,
        feature_params=feature_params,
        extensions=extensions,
    )


def _process_images(config: CliConfig) -> List[Dict[str, Any]]:
    """Process images according to the collector configuration."""

    LOGGER.info("Scanning %s for images", config.image_directory)
    images = _collect_images(config.image_directory, config.extensions)
    LOGGER.info("Found %d image(s) to process", len(images))
    LOGGER.info("Running %s detector", config.feature_type)

    results: List[Dict[str, Any]] = []
    for image_path in iter_with_progress(images):
        relative_name = _relative_path(config.image_directory, image_path)
        LOGGER.debug("Processing image %s", relative_name)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        if config.feature_type == "charuco":
            detection = _detect_charuco(image, config.feature_params)
            LOGGER.debug("Detected %d corner(s) in %s", detection["count"], relative_name)
        else:
            detection = _detect_keypoints(image, config.feature_type, config.feature_params)
            LOGGER.debug(
                "Detected %d keypoint(s) with %s in %s",
                detection["count"],
                config.feature_type,
                relative_name,
            )
        detection["file"] = relative_name
        results.append(detection)
    return results


def run_from_config(config_path: Path) -> Dict[str, Any]:
    """Run the CLI pipeline using the specified config path."""

    config = _resolve_config(config_path)
    detections = _process_images(config)

    payload: Dict[str, Any] = {
        "image_directory": str(config.image_directory),
        "feature_type": config.feature_type,
        "algo_version": algo_version(config.feature_type),
        "params_hash": params_hash(config.feature_params),
        "images": detections,
    }

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing detection results to %s", config.output_file)
    with config.output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    LOGGER.info("Completed detection for %d image(s)", len(detections))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (exposed for testing)."""

    parser = argparse.ArgumentParser(description="Run feature detection from a JSON config.")
    parser.add_argument("config", type=str, help="Path to the configuration JSON file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point used by the CLI script."""

    parser = build_parser()
    args = parser.parse_args(argv)

    LOGGER.configure(args.log_level)

    config_path = Path(args.config).resolve()
    run_from_config(config_path)


if __name__ == "__main__":  # pragma: no cover - manual executions only
    main()
