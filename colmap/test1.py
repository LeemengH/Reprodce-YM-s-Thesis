import argparse
import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def read_first_camera(cameras_txt_path):
    with open(cameras_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            return model, width, height, params
    raise RuntimeError(f"No camera found in {cameras_txt_path}")

def read_images(images_txt_path):
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
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        poses.append((qw, qx, qy, qz, tx, ty, tz, name))
        if i < len(lines):
            i += 1
    return poses

def qvec_to_rotmat(qw, qx, qy, qz):
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def parse_intrinsics(model, width, height, params):
    k1 = 0.0
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        fx = fy = f
    elif model == "RADIAL":
        f, cx, cy, k1, _k2 = params
        fx = fy = f
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, _k2, _p1, _p2 = params
    else:
        raise NotImplementedError(f"Camera model '{model}' not supported yet.")
    return fx, fy, cx, cy, k1, width, height

def apply_k1(x, y, k1):
    if k1 == 0.0:
        return x, y
    r2 = x*x + y*y
    s = 1.0 + k1*r2
    return x*s, y*s

def project_point(P_c, fx, fy, cx, cy, k1, W, H, depth_axis="z"):
    """
    depth_axis=z: u=fx*(X/Z)+cx, v=fy*(Y/Z)+cy
    depth_axis=y: treat Y as forward/depth -> u=fx*(X/Y)+cx, v=fy*(Z/Y)+cy
    """
    if depth_axis == "z":
        d = P_c[2]
        if d <= 1e-9:
            return None
        x = P_c[0] / d
        y = P_c[1] / d
    else:
        d = P_c[1]
        if d <= 1e-9:
            return None
        x = P_c[0] / d
        y = P_c[2] / d

    x, y = apply_k1(x, y, k1)
    u = fx*x + cx
    v = fy*y + cy
    if 0 <= u < W and 0 <= v < H:
        return int(round(u)), int(round(v))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse_txt", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--keyframe", required=True, help="path to keyframe image (abs or rel)")
    ap.add_argument("--d", type=float, default=2.0)
    ap.add_argument("--forward_axis", choices=["y", "z"], default="y")
    ap.add_argument("--depth_axis", choices=["y", "z"], default="y")
    ap.add_argument("--stride", type=int, default=1, help="sample every N future frames")
    ap.add_argument("--horizon", type=int, default=12, help="number of trajectory points to output")
    ap.add_argument("--output", default="traj.png")
    ap.add_argument("--dump_json", default="", help="if set, write gt_traj json to this path")
    args = ap.parse_args()

    cameras_txt = os.path.join(args.sparse_txt, "cameras.txt")
    images_txt = os.path.join(args.sparse_txt, "images.txt")

    model, W, H, params = read_first_camera(cameras_txt)
    fx, fy, cx, cy, k1, W, H = parse_intrinsics(model, W, H, params)
    poses = read_images(images_txt)

    keyframe_path = os.path.abspath(args.keyframe)
    key_base = os.path.basename(keyframe_path)

    key_idx = None
    key_name_in_model = None
    for idx, p in enumerate(poses):
        if os.path.basename(p[7]) == key_base:
            key_idx = idx
            key_name_in_model = p[7]
            break
    if key_idx is None:
        raise FileNotFoundError(f"Keyframe '{key_base}' not found in images.txt")

    print(f"[INFO] Camera model: {model}, W={W}, H={H}, k1={k1}")
    print(f"[INFO] Using keyframe index={key_idx}, name_in_images_txt='{key_name_in_model}'")

    # forward axis vector in camera frame
    forward_cam = np.array([0.0, 1.0, 0.0]) if args.forward_axis == "y" else np.array([0.0, 0.0, 1.0])

    # compute P_i in world for all frames
    points_world = []
    for (qw, qx, qy, qz, tx, ty, tz, _name) in poses:
        Rmat = qvec_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        C = -Rmat.T @ t
        forward_world = Rmat.T @ forward_cam
        #P = C + args.d * forward_world
        P = C
        points_world.append(P)
    points_world = np.asarray(points_world, dtype=np.float64)

    # keyframe pose (world->cam)
    qw, qx, qy, qz, tx, ty, tz, _ = poses[key_idx]
    R_ref = qvec_to_rotmat(qw, qx, qy, qz)
    t_ref = np.array([tx, ty, tz], dtype=np.float64)

    # project with stride + horizon
    uv_list = []
    total = 0
    front_like = 0  # passes depth>0
    inside = 0

    future_indices = list(range(key_idx, len(points_world), args.stride))
    for j, idx in enumerate(future_indices):
        if len(uv_list) >= args.horizon:
            break
        total += 1
        P_c = R_ref @ points_world[idx] + t_ref
        uv = project_point(P_c, fx, fy, cx, cy, k1, W, H, depth_axis=args.depth_axis)
        if args.depth_axis == "z":
            if P_c[2] > 1e-9:
                front_like += 1
        else:
            if P_c[1] > 1e-9:
                front_like += 1

        if uv is not None:
            inside += 1
            uv_list.append(uv)

    print(f"[DEBUG] sampled={total}, depth_positive={front_like}, inside_image={inside}, uv_list={len(uv_list)}")
    print(uv_list)

    # overlay
    img = cv2.imread(keyframe_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {keyframe_path}")

    for (u, v) in uv_list:
        cv2.circle(img, (u, v), 3, (0, 0, 255), -1)

    cv2.imwrite(args.output, img)
    print(f"[INFO] Saved {args.output}")

    if args.dump_json:
        out = {"gt_traj": [[int(u), int(v)] for (u, v) in uv_list]}
        with open(args.dump_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Saved {args.dump_json}")

if __name__ == "__main__":
    main()
