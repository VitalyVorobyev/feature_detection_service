import cv2, numpy as np

def _odd(n: int) -> int:  # helper
    return int(n) if int(n) % 2 else int(n) + 1

def detect_charuco(gray: np.ndarray, squares_x=21, squares_y=21,
                   square_length=1.0, marker_length=0.5,
                   dictionary="DICT_5X5_1000",
                   return_overlay=False):
    """
    Returns: points (N,2) float32, ids (N,), object_points (N,3), overlay (or None).
    object_points are the coordinates of the detected corners in the
    board's local coordinate system (z=0 plane).
    """
    assert gray.ndim == 2 and gray.dtype == np.uint8
    H, W = gray.shape[:2]
    perim = 2*(W+H)

    # --- pick dictionary (ensure it matches your printed board!) ---
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"Bad dictionary {dictionary}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))

    # --- expected scale in pixels ---
    sq_px = min(W/float(squares_x), H/float(squares_y))
    r = float(marker_length)/float(square_length) if square_length > 0 else 0.5
    marker_px = max(1.0, sq_px * r)

    # ArUco default threshold rejects small markers; lower it based on expected size.
    # side must satisfy: 4*side >= minMarkerPerimeterRate * perim
    min_perim_rate = max(0.005, 0.6 * (4.0 * marker_px / perim))  # 60% of theoretical bound

    # Adaptive threshold windows relative to marker scale
    win_min = _odd(max(5, int(round(marker_px * 0.6))))
    win_max = _odd(max(win_min+2, int(round(marker_px * 2.5))))
    win_step = 2 if (win_max - win_min) > 12 else 1

    # Corner refinement window ~15% of marker size (odd)
    cr_win = _odd(max(3, int(round(marker_px * 0.15))))

    # --- parameters (new API names) ---
    detpar = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") \
             else cv2.aruco.DetectorParameters_create()

    detpar.adaptiveThreshWinSizeMin = win_min
    detpar.adaptiveThreshWinSizeMax = win_max
    detpar.adaptiveThreshWinSizeStep = win_step
    detpar.adaptiveThreshConstant = 5
    detpar.polygonalApproxAccuracyRate = 0.01

    detpar.minMarkerPerimeterRate = float(min_perim_rate)
    detpar.maxMarkerPerimeterRate = 4.0
    detpar.minCornerDistanceRate = 0.005
    detpar.minDistanceToBorder = 1
    detpar.markerBorderBits = 1
    detpar.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    detpar.cornerRefinementWinSize = cr_win
    detpar.cornerRefinementMinAccuracy = 0.05
    if hasattr(detpar, "detectInvertedMarker"):
        detpar.detectInvertedMarker = True  # helps if image/lighting flips contrast
    if hasattr(detpar, "useAruco3Detection"):
        detpar.useAruco3Detection = True

    chpar = cv2.aruco.CharucoParameters() if hasattr(cv2.aruco, "CharucoParameters") else None
    if chpar is not None:
        chpar.minMarkers = 2
        chpar.tryRefineMarkers = True

    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

    def _run(gray_img):
        # prefer modern CharucoDetector when available
        if hasattr(cv2.aruco, "CharucoDetector"):
            refpar = cv2.aruco.RefineParameters()
            detector = cv2.aruco.CharucoDetector(board, charucoParams=chpar,
                                                 detectorParams=detpar, refineParams=refpar)
            ch_corners, ch_ids, _, _ = detector.detectBoard(gray_img)
        else:
            # legacy path
            corners, ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=detpar)
            ch_corners, ch_ids = None, None
            if ids is not None and len(ids) > 0:
                res = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_img, board)
                if res is not None and len(res) >= 3:
                    ch_corners, ch_ids = res[1], res[2]
        return ch_corners, ch_ids

    # pass 1
    ch_corners, ch_ids = _run(gray)

    # fallback pass: loosen size threshold further and try again
    if ch_ids is None or len(ch_ids) == 0:
        detpar.minMarkerPerimeterRate = max(0.003, min_perim_rate * 0.5)
        ch_corners, ch_ids = _run(gray)

    if ch_ids is None or len(ch_ids) == 0:
        # last resort: modest upscaling (helps tiny markers)
        up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        ch_corners, ch_ids = _run(up)
        if ch_ids is not None and len(ch_ids) > 0:
            # map back to original coords
            ch_corners = ch_corners / 1.5

    if ch_ids is None or len(ch_ids) == 0:
        return (np.empty((0,2), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,3), np.float32),
                (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if return_overlay else None))

    # subpix refine on original-res image
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    try:
        cv2.cornerSubPix(gray, ch_corners, (cr_win, cr_win), (-1, -1), term)
    except Exception:
        pass

    pts = ch_corners.reshape(-1, 2).astype(np.float32)
    ids = ch_ids.reshape(-1).astype(np.int32)
    objpts = board.getChessboardCorners()[ids].astype(np.float32)

    overlay = None
    if return_overlay:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        try:
            cv2.aruco.drawDetectedCornersCharuco(overlay, pts.reshape(-1,1,2), ids)
        except Exception:
            for (x,y) in pts:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0,255,0), -1)

    return pts, ids, objpts, overlay
