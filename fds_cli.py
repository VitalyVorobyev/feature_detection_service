import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fds import algo_version, algos, make_detector, params_hash
from features import detect_charuco

try:
    import spdlog  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spdlog = None

try:
    from tqdm import tqdm  # type: ignore

    _TQDM_ENABLED = True
except ImportError:  # pragma: no cover - optional dependency
    _TQDM_ENABLED = False

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


class _CompatLogger:
    """Small adapter that hides backend differences (logging vs spdlog)."""

    def __init__(self, backend):
        self._backend = backend

    def _emit(self, method: str, msg: str, *args: Any) -> None:
        if args:
            msg = msg % args
        logger_fn = getattr(self._backend, method, None)
        if logger_fn is None and method == "warning":
            logger_fn = getattr(self._backend, "warn")
        if logger_fn is None:
            return
        logger_fn(msg)

    def info(self, msg: str, *args: Any) -> None:
        self._emit("info", msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._emit("debug", msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._emit("warning", msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._emit("error", msg, *args)

    def critical(self, msg: str, *args: Any) -> None:
        self._emit("critical", msg, *args)


logger = _CompatLogger(logging.getLogger("fds_cli"))


def _setup_logger(level_name: str) -> None:
    """Configure global logger using spdlog when available."""

    global logger
    level_upper = level_name.upper()

    if spdlog is not None:
        backend = spdlog.ConsoleLogger("fds-cli")
        level = getattr(getattr(spdlog, "LogLevel"), level_upper, None)
        if level is None:
            level = getattr(getattr(spdlog, "LogLevel"), "INFO")
        backend.set_level(level)
        logger = _CompatLogger(backend)
    else:
        level = getattr(logging, level_upper, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        backend = logging.getLogger("fds_cli")
        backend.setLevel(level)
        logger = _CompatLogger(backend)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")

    required_fields = ["image_directory", "feature_type", "feature_params", "output_file"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Config missing required fields: {', '.join(missing)}")

    return data


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _collect_images(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {directory}")

    exts = extensions or [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    images = [p for p in sorted(directory.glob("**/*")) if p.is_file() and p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images found in {directory} with extensions {sorted(exts)}")
    return images


def _serialize_keypoints(keypoints: List[cv2.KeyPoint]) -> List[Dict[str, Any]]:
    serialized = []
    for kp in keypoints:
        serialized.append(
            {
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave),
                "class_id": int(getattr(kp, "class_id", -1)),
            }
        )
    return serialized


def _relative_path(root: Path, file_path: Path) -> str:
    try:
        return str(file_path.relative_to(root))
    except ValueError:
        return str(file_path)


def _detect_keypoints(image: np.ndarray, algo_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    detector = make_detector(algo_name, params)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    keypoints = keypoints or []

    serialized_keypoints = _serialize_keypoints(keypoints)
    if descriptors is None:
        serialized_descriptors: List[List[float]] = []
    else:
        dtype = np.float32 if algo_name == "sift" else np.uint8
        serialized_descriptors = descriptors.astype(dtype).tolist()

    return {
        "count": len(serialized_keypoints),
        "keypoints": serialized_keypoints,
        "descriptors": serialized_descriptors,
    }


def _detect_charuco(image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points, ids, obj_points, _ = detect_charuco(gray, return_overlay=False, **params)
    if points is None or ids is None or obj_points is None:
        points_list: List[Dict[str, Any]] = []
    else:
        points_list = []
        for idx, pt, obj in zip(ids.flatten(), points, obj_points):
            points_list.append(
                {
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "id": int(idx),
                    "local_x": float(obj[0]),
                    "local_y": float(obj[1]),
                    "local_z": float(obj[2]),
                }
            )

    return {
        "count": len(points_list),
        "points": points_list,
    }


def _is_supported_algo(name: str) -> bool:
    supported = {entry["name"] for entry in algos()}
    supported.add("charuco")
    return name in supported


def run_from_config(config_path: Path) -> Dict[str, Any]:
    logger.info("Loading configuration from %s", config_path)
    config_data = _load_config(config_path)
    base_dir = config_path.parent

    image_directory = _resolve_path(base_dir, config_data["image_directory"])
    output_file = _resolve_path(base_dir, config_data["output_file"])
    feature_type = str(config_data["feature_type"]).lower()
    feature_params = config_data.get("feature_params", {})
    extensions = config_data.get("extensions")

    logger.info("Resolved image directory: %s", image_directory)
    logger.info("Resolved output file: %s", output_file)
    logger.info("Using feature type '%s'", feature_type)
    if feature_params:
        logger.debug("Feature parameters: %s", feature_params)

    if not _is_supported_algo(feature_type):
        raise ValueError(f"Unsupported feature type '{feature_type}'")

    normalized_exts = None
    if extensions is not None:
        normalized_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    exts_for_log = normalized_exts or {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    logger.info("Scanning %s for images with extensions %s", image_directory, sorted(exts_for_log))

    images = _collect_images(image_directory, extensions)
    logger.info("Found %d image(s) to process", len(images))
    logger.info("Running %s detector", feature_type)

    progress_bar = None
    if _TQDM_ENABLED and len(images) > 1:
        progress_bar = tqdm(images, desc="Processing images", unit="image")
        iterable = progress_bar
    else:
        iterable = images

    results = []
    for image_path in iterable:
        rel_path = _relative_path(image_directory, image_path)
        logger.debug("Processing image %s", rel_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        if feature_type == "charuco":
            detection = _detect_charuco(image, feature_params)
            logger.debug("Detected %d corner(s) in %s", detection["count"], rel_path)
        else:
            detection = _detect_keypoints(image, feature_type, feature_params)
            logger.debug(
                "Detected %d keypoint(s) with %s in %s",
                detection["count"],
                feature_type,
                rel_path,
            )

        if progress_bar is not None:
            progress_bar.set_postfix(count=detection["count"], file=rel_path)

        detection["file"] = rel_path
        results.append(detection)

    if progress_bar is not None:
        progress_bar.close()

    payload: Dict[str, Any] = {
        "image_directory": str(image_directory),
        "feature_type": feature_type,
        "algo_version": algo_version(feature_type),
        "params_hash": params_hash(feature_params),
        "images": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing detection results to %s", output_file)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Completed detection for %d image(s)", len(results))
    return payload


def main(argv: Optional[List[str]] = None) -> None:
    """ Run me """
    parser = argparse.ArgumentParser(description="Run feature detection from a JSON config.")
    parser.add_argument("config", type=str, help="Path to the configuration JSON file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    _setup_logger(args.log_level)

    config_path = Path(args.config).resolve()
    run_from_config(config_path)


if __name__ == "__main__":
    main()
