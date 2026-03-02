import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os

# === 修改這裡 ===
SPARSE_TXT = "./sparse_txt"
IMAGE_FOLDER = "./dataset"
KEYFRAME_INDEX = 72  # 第20張（0-based）
D = 1.0  # COLMAP scale 下的 forward distance
# =================

def read_cameras(path):
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            break
    fx, cx, cy, k1 = params
    return fx, fx, cx, cy, k1, width, height

def read_images(path):
    poses = []
    with open(path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith("#"):
            continue
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        poses.append((qw, qx, qy, qz, tx, ty, tz, name))
    return poses

def qvec2rotmat(qw, qx, qy, qz):
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

# === 讀資料 ===
fx, fy, cx, cy, k1, W, H = read_cameras(os.path.join(SPARSE_TXT, "cameras.txt"))
poses = read_images(os.path.join(SPARSE_TXT, "images.txt"))

# === 算所有相機中心 + forward 3D點 ===
points_world = []

for qw, qx, qy, qz, tx, ty, tz, _ in poses:
    Rmat = qvec2rotmat(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    
    C = -Rmat.T @ t  # camera center
    
    forward_cam = np.array([0,1,0])
    forward_world = Rmat.T @ forward_cam
    
    P = C + D * forward_world
    points_world.append(P)

points_world = np.array(points_world)

# === Keyframe pose ===
qw, qx, qy, qz, tx, ty, tz, name = poses[KEYFRAME_INDEX]
R_ref = qvec2rotmat(qw, qx, qy, qz)
t_ref = np.array([tx, ty, tz])

# === 投影 ===
uv_list = []

for P in points_world[KEYFRAME_INDEX:]:
    Pc = R_ref @ P + t_ref
    
    if Pc[2] <= 0:
        continue
        
    x = Pc[0] / Pc[2]
    y = Pc[1] / Pc[2]
    
    r2 = x*x + y*y
    x = x * (1 + k1*r2)
    y = y * (1 + k1*r2)
    
    u = fx*x + cx
    v = fy*y + cy
    
    if 0 <= u < W and 0 <= v < H:
        uv_list.append((int(u), int(v)))

# === 畫圖 ===
img = cv2.imread(os.path.join(IMAGE_FOLDER, name))

for (u,v) in uv_list:
    cv2.circle(img, (u,v), 3, (0,0,255), -1)
print(uv_list)
cv2.imwrite("trajectory_keyframe20.jpg", img)
print("Saved trajectory_keyframe20.jpg")
