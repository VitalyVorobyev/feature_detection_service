# fds/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests, cv2, numpy as np, json, hashlib, base64, os

from features import detect_charuco

ISS_URL = os.environ.get("ISS_URL", "http://localhost:8000")
app = FastAPI(title="Feature Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,  # set False if you don’t use cookies/auth
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    # allow_headers can be "*", but if you prefer explicit list, include the ones you use:
    allow_headers=["*"],  # or ["Authorization","Content-Type","X-Request-Id","Range"]
    # expose headers you need to read from JS (useful for Range/ETag):
    expose_headers=["Content-Range", "ETag"],
    max_age=86400,
)

class DetectReq(BaseModel):
    image_id: str
    algo: str = Field(description="orb|sift|akaze|brisk")
    params: dict = {}
    return_overlay: bool = False


class PatternDetectReq(BaseModel):
    image_id: str
    pattern: str = Field(description="charuco|circle_grid|chessboard|apriltag")
    params: dict = {}
    return_overlay: bool = False

def load_image_from_iss(image_id: str) -> np.ndarray:
    r = requests.get(f"{ISS_URL}/images/{image_id}", timeout=30)
    if r.status_code != 200: raise HTTPException(404, "image not found")
    buf = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "invalid image bytes")
    print(f"Loaded image {image_id} from ISS {ISS_URL}, shape={img.shape}")
    return img

def algo_version(algo: str) -> str:
    return f"opencv-{cv2.__version__}"

def make_detector(algo: str, params: dict):
    if algo == "orb":
        return cv2.ORB_create(
            nfeatures=int(params.get("n_features", 2000)),
            scaleFactor=float(params.get("scaleFactor", 1.2)),
            nlevels=int(params.get("nlevels", 8)))
    if algo == "sift":
        return cv2.SIFT_create(
            nfeatures=int(params.get("n_features", 2000)))
    if algo == "akaze":
        return cv2.AKAZE_create()
    if algo == "brisk":
        return cv2.BRISK_create()
    raise HTTPException(400, f"unknown algo {algo}")

def params_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def encode_png(img: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def encode_overlay(img: np.ndarray, kpts):
    overlay = img.copy()
    cv2.drawKeypoints(img, kpts, overlay, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return encode_png(overlay)

@app.post("/detect")
def detect(req: DetectReq):
    img = load_image_from_iss(req.image_id)
    det = make_detector(req.algo, req.params)
    kpts, desc = det.detectAndCompute(img, None)
    count = 0 if kpts is None else len(kpts)
    if count == 0:
        summary = []
        desc = np.zeros((0, 32), dtype=np.uint8)
    else:
        # pack keypoints into (N,7) float32/int32
        kp_arr = np.zeros((count, 7), dtype=np.float32)
        for i, k in enumerate(kpts):
            kp_arr[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, float(k.octave), float(getattr(k, "class_id", -1)))
        if desc is None:
            desc = np.zeros((0, 32), dtype=np.uint8)
        # store as npz in ISS
        import io as _io
        bio = _io.BytesIO()
        np.savez_compressed(bio, keypoints=kp_arr, descriptors=desc.astype(np.float32 if req.algo=="sift" else np.uint8))
        files = {"file": ("features.npz", bio.getvalue(), "application/octet-stream")}
        meta = json.dumps({"image_id": req.image_id, "algo": req.algo})
        r = requests.post(f"{ISS_URL}/artifacts?kind=features&meta={meta}", files=files, timeout=30)
        r.raise_for_status()
        features_artifact_id = r.json()["artifact_id"]
        # compact summary for UI (cap to 1000)
        cap = min(1000, count)
        summary = [{"x": float(k.pt[0]), "y": float(k.pt[1]), "size": float(k.size),
                    "angle": float(k.angle), "resp": float(k.response), "octave": int(k.octave)}
                   for k in kpts[:cap]]
    resp = {
        "image_id": req.image_id,
        "algo": req.algo,
        "algo_version": algo_version(req.algo),
        "params_hash": params_hash(req.params),
        "count": count,
        "summary": summary,
        "features_artifact_id": r.json()["artifact_id"] if count>0 else None
    }
    if req.return_overlay:
        resp["overlay_png"] = encode_overlay(img, kpts or [])
    return resp


@app.post("/detect_pattern")
def detect_pattern(req: PatternDetectReq):
    img = load_image_from_iss(req.image_id)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    overlay = img.copy()
    pattern = req.pattern
    params = req.params

    def use_detect_charuco(gray, overlay, params):
        corners, ids, objpts, ch_overlay = detect_charuco(gray, return_overlay=True, **params)
        if ch_overlay is not None:
            overlay[:]=ch_overlay
        points = []
        for i, c, o in zip(ids.flatten(), corners, objpts):
            points.append({
                "x": float(c[0]),
                "y": float(c[1]),
                "id": int(i),
                "local_x": float(o[0]),
                "local_y": float(o[1]),
                "local_z": float(o[2]),
            })
        return points

    def detect_circle_grid(gray, overlay, params):
        rows = int(params.get("rows", 4))
        cols = int(params.get("cols", 5))
        flags = cv2.CALIB_CB_SYMMETRIC_GRID if params.get("symmetric", True) else cv2.CALIB_CB_ASYMMETRIC_GRID
        found, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
        points = []
        if found:
            cv2.drawChessboardCorners(overlay, (cols, rows), centers, found)
            for idx, p in enumerate(centers):
                col = idx % cols
                row = idx // cols
                points.append({
                    "x": float(p[0][0]),
                    "y": float(p[0][1]),
                    "local_x": float(col),
                    "local_y": float(row),
                })
        return points

    def detect_chessboard(gray, overlay, params):
        rows = int(params.get("rows", 7))
        cols = int(params.get("cols", 7))
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        points = []
        if found:
            cv2.drawChessboardCorners(overlay, (cols, rows), corners, found)
            for idx, p in enumerate(corners):
                col = idx % cols
                row = idx // cols
                points.append({
                    "x": float(p[0][0]),
                    "y": float(p[0][1]),
                    "local_x": float(col),
                    "local_y": float(row),
                })
        return points

    def detect_apriltag(gray, overlay, params):
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, params.get("dictionary", "DICT_APRILTAG_36h11")))
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        points = []
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                center = c[0].mean(axis=0)
                points.append({
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "id": int(i),
                    "local_x": 0.0,
                    "local_y": 0.0,
                })
        return points

    pattern_funcs = {
        "charuco": use_detect_charuco,
        "circle_grid": detect_circle_grid,
        "chessboard": detect_chessboard,
        "apriltag": detect_apriltag,
    }

    if pattern not in pattern_funcs:
        raise HTTPException(400, f"unknown pattern {pattern}")

    points = pattern_funcs[pattern](gray, overlay, params)
    if points is None:
        points = []

    resp = {
        "image_id": req.image_id,
        "pattern": pattern,
        "algo_version": algo_version(pattern),
        "params_hash": params_hash(params),
        "count": len(points),
        "points": points,
    }
    if req.return_overlay:
        resp["overlay_png"] = encode_png(overlay)
    return resp

@app.get("/algos")
def algos():
    return [
        {"name":"orb","params":{"n_features":"int","scaleFactor":"float","nlevels":"int"}},
        {"name":"sift","params":{"n_features":"int"}},
        {"name":"akaze","params":{}},
        {"name":"brisk","params":{}},
    ]


@app.get("/patterns")
def patterns():
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
        {
            "name": "chessboard",
            "params": {"rows": "int", "cols": "int"},
        },
        {
            "name": "apriltag",
            "params": {"dictionary": "str"},
        },
    ]

@app.get("/healthz")
def health(): return {"ok": True}
