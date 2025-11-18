import os
import cv2
import numpy as np
import open3d as o3d

# ================== ПАРАМЕТРЫ ==================
VIDEO_PATH   = "IMG_0904.MOV"   # ← путь к твоему видео
OUTPUT_DIR   = "output"
PLY_PATH     = os.path.join(OUTPUT_DIR, "box_sfm_color.ply")

FRAME_STRIDE = 1      # 1 = каждый кадр, можно 2 или 3 если долго
DOWNSCALE    = 0.6    # уменьшить разрешение для ускорения
MIN_MATCHES  = 80
FOV_DEG      = 70.0   # угол обзора камеры (iPhone ~70)
TOPK_POINTS  = 150000

# ================== ФУНКЦИИ ==================
def K_from_fov(w, h, fov_deg):
    f = (0.5 * w) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0,      1 ]], dtype=np.float64)

def triangulate(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T
    return X

def blue_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 60, 40])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    return mask

# ================== ПОДГОТОВКА ==================
os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть {VIDEO_PATH}")

try:
    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
except:
    sift = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

prev_kp, prev_des, prev_rgb, prev_mask = None, None, None, None
