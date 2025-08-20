# fds/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests, cv2, numpy as np, json, hashlib, base64, os

ISS_URL = os.environ.get("ISS_URL", "http://localhost:8081")
app = FastAPI(title="Feature Detection Service")

class DetectReq(BaseModel):
    image_id: str
    algo: str = Field(description="orb|sift|akaze|brisk")
    params: dict = {}
    return_overlay: bool = False

def load_image_from_iss(image_id: str) -> np.ndarray:
    r = requests.get(f"{ISS_URL}/images/{image_id}", timeout=30)
    if r.status_code != 200: raise HTTPException(404, "image not found")
    buf = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "invalid image bytes")
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

def encode_overlay(img: np.ndarray, kpts):
    overlay = img.copy()
    cv2.drawKeypoints(img, kpts, overlay, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ok, png = cv2.imencode(".png", overlay); assert ok
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()

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

@app.get("/algos")
def algos():
    return [
        {"name":"orb","params":{"n_features":"int","scaleFactor":"float","nlevels":"int"}},
        {"name":"sift","params":{"n_features":"int"}},
        {"name":"akaze","params":{}},
        {"name":"brisk","params":{}},
    ]

@app.get("/healthz")
def health(): return {"ok": True}
