import argparse
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def read_first_camera(cameras_txt_path):
    """
    Read the first non-comment camera line from COLMAP cameras.txt.
    Supports SIMPLE_PINHOLE / PINHOLE / SIMPLE_RADIAL / RADIAL / OPENCV partially.
    For reproduction, we mainly use SIMPLE_RADIAL or SIMPLE_PINHOLE.
    """
    with open(cameras_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            # Return raw; parse later by model
            return model, width, height, params

    raise RuntimeError(f"No camera found in {cameras_txt_path}")

def read_images(images_txt_path):
    """
    Parse COLMAP images.txt into list of poses:
      (qw,qx,qy,qz, tx,ty,tz, name)
    COLMAP convention: X_cam = R * X_world + t  (world->cam)
    """
    poses = []
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        # IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        poses.append((qw, qx, qy, qz, tx, ty, tz, name))

        # Skip the next line (2D points)
        if i < len(lines):
            i += 1

    return poses

def qvec_to_rotmat(qw, qx, qy, qz):
    # scipy expects (x,y,z,w)
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def parse_intrinsics(model, width, height, params):
    """
    Return fx, fy, cx, cy, k1 (k1 may be 0 if not available)
    """
    k1 = 0.0

    if model == "SIMPLE_PINHOLE":
        # params: f, cx, cy
        f, cx, cy = params
        fx = fy = f
    elif model == "PINHOLE":
        # params: fx, fy, cx, cy
        fx, fy, cx, cy = params
    elif model == "SIMPLE_RADIAL":
        # params: f, cx, cy, k1
        f, cx, cy, k1 = params
        fx = fy = f
    elif model == "RADIAL":
        # params: f, cx, cy, k1, k2
        f, cx, cy, k1, _k2 = params
        fx = fy = f
    elif model == "OPENCV":
        # params: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy, k1, _k2, _p1, _p2 = params
        # 這裡為了復刻學長的 k1-only，先只用 k1
        # 若你需要完整畸變，我可以再幫你補 OPENCV 畸變投影
    else:
        raise NotImplementedError(
            f"Camera model '{model}' not supported in this minimal script. "
            f"Please paste cameras.txt first line and I’ll extend it."
        )

    return fx, fy, cx, cy, k1, width, height

def apply_simple_radial_distortion(x, y, k1):
    if k1 == 0.0:
        return x, y
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2
    return x * factor, y * factor
    
def project_point(P_c, fx, fy, cx, cy, k1, W, H, depth_axis="z"):
    # Choose depth
    if depth_axis == "z":
        depth = P_c[2]
        if depth <= 1e-9:
            return None
        x = P_c[0] / depth
        y = P_c[1] / depth
    else:  # y-forward convention
        depth = P_c[1]
        if depth <= 1e-9:
            return None
        # If Y is forward, a common choice is x = X/Y, y = Z/Y (swap to match "vertical")
        x = P_c[0] / depth
        y = P_c[2] / depth

    # k1-only distortion (same as before)
    r2 = x*x + y*y
    x = x * (1 + k1*r2)
    y = y * (1 + k1*r2)

    u = fx * x + cx
    v = fy * y + cy

    if 0 <= u < W and 0 <= v < H:
        return int(round(u)), int(round(v))
    return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse_txt", required=True, help="Path to COLMAP TXT model folder (contains cameras.txt/images.txt)")
    ap.add_argument("--image_root", required=True, help="Dataset root that contains the relative paths in images.txt")
    ap.add_argument("--keyframe", required=True, help="ABSOLUTE path to the keyframe image file")
    ap.add_argument("--d", type=float, default=1.0, help="Forward distance in COLMAP scale (default: 1.0)")
    ap.add_argument("--forward_axis", choices=["y", "z"], default="z", help="Forward axis in camera frame (default: z)")
    ap.add_argument("--depth_axis", choices=["y", "z"], default="z",
                help="Which axis is treated as depth during projection (default: z)")
    ap.add_argument("--output", default="trajectory_keyframe.png", help="Output overlay image path")
    args = ap.parse_args()

    cameras_txt = os.path.join(args.sparse_txt, "cameras.txt")
    images_txt = os.path.join(args.sparse_txt, "images.txt")

    model, W, H, params = read_first_camera(cameras_txt)
    fx, fy, cx, cy, k1, W, H = parse_intrinsics(model, W, H, params)

    poses = read_images(images_txt)
    if len(poses) == 0:
        raise RuntimeError("No images parsed from images.txt")

    keyframe_abs = os.path.abspath(args.keyframe)
    keyframe_base = os.path.basename(keyframe_abs)

    # Find keyframe by matching basename against images.txt names
    key_idx = None
    key_name_in_model = None
    for idx, p in enumerate(poses):
        name = p[7]  # e.g. images/000072.png
        if os.path.basename(name) == keyframe_base:
            key_idx = idx
            key_name_in_model = name
            break

    if key_idx is None:
        # show a hint: closest names
        sample_names = [os.path.basename(p[7]) for p in poses[:10]]
        raise FileNotFoundError(
            f"Keyframe basename '{keyframe_base}' not found in images.txt. "
            f"First 10 basenames: {sample_names}"
        )

    print(f"[INFO] Camera model: {model}, W={W}, H={H}, k1={k1}")
    print(f"[INFO] Using keyframe index={key_idx}, name_in_images_txt='{key_name_in_model}'")

    # Compute world points P_i = C_i + d * forward_world for all frames
    points_world = []
    if args.forward_axis == "y":
        forward_cam = np.array([0.0, 1.0, 0.0])
    else:
        forward_cam = np.array([0.0, 0.0, 1.0])

    for (qw, qx, qy, qz, tx, ty, tz, _name) in poses:
        Rmat = qvec_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)

        # camera center in world
        C = -Rmat.T @ t

        # forward in world
        forward_world = Rmat.T @ forward_cam

        P = C + args.d * forward_world
        points_world.append(P)

    points_world = np.asarray(points_world, dtype=np.float64)

    # Keyframe pose (world->cam)
    qw, qx, qy, qz, tx, ty, tz, _ = poses[key_idx]
    R_ref = qvec_to_rotmat(qw, qx, qy, qz)
    t_ref = np.array([tx, ty, tz], dtype=np.float64)

    # Project future points (from key_idx onward) into keyframe image
    uv_list = []
    total = 0
    front = 0
    in_img = 0

    for P_w in points_world[key_idx:]:
        total += 1
        P_c = R_ref @ P_w + t_ref  # world->keyframe_cam

        # Choose projection convention
        if args.depth_axis == "z":
            depth = P_c[2]
            if depth <= 1e-9:
                continue
            front += 1

            x = P_c[0] / depth
            y = P_c[1] / depth

        else:  # args.depth_axis == "y"  (Y-forward convention)
            depth = P_c[1]
            if depth <= 1e-9:
                continue
            front += 1

            # If Y is forward, common choice: x = X / Y, y = Z / Y
            x = P_c[0] / depth
            y = P_c[2] / depth

        x, y = apply_simple_radial_distortion(x, y, k1)

        u = fx * x + cx
        v = fy * y + cy

        if 0 <= u < W and 0 <= v < H:
            in_img += 1
            uv_list.append((int(round(u)), int(round(v))))

    print(f"[DEBUG] total future pts={total}, in front={front}, inside image={in_img}, uv_list={len(uv_list)}")

    # Load keyframe image from absolute path (what you asked)
    img = cv2.imread(keyframe_abs, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read keyframe image at: {keyframe_abs}")
    H_img, W_img = img.shape[:2]

    # W, H 是 cameras.txt 的尺寸
    scale_x = W_img / float(W)
    scale_y = H_img / float(H)

    # 在你 append uv_list 前，把 u,v scale 成 overlay 圖的座標系
    u = u * scale_x
    v = v * scale_y
    # Overlay points
    for (u, v) in uv_list:
        cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
    print(uv_list)
    cv2.imwrite(args.output, img)
    print(f"[INFO] Saved {args.output}")

if __name__ == "__main__":
    main()
