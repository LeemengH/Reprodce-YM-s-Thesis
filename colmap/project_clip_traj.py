#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
project_clip_traj.py

- Reads COLMAP sparse model exported as text: cameras.txt + images.txt
- Pick one keyframe in a clip, generate future lookahead camera positions (future-only)
- Project those 3D lookahead points onto the keyframe image (with SIMPLE_RADIAL k1)
- Supports resizing the keyframe image AND synchronously scaling K (fx,fy,cx,cy)
- Optionally overlays GT trajectory from titan_train.json (robust parsing)

Example:
python project_clip_traj.py \
  --cameras_txt ./sparse_txt/cameras.txt \
  --images_txt  ./sparse_txt/images.txt \
  --image_root  ./dataset/clip_1 \
  --keyframe    images/000438.png \
  --gt_json     ~/Downloads/ym_reproduce/exclude_no_gt/titan_train.json \
  --clip_id     clip_1 \
  --lookahead_d 2.0 \
  --forward_axis y \
  --resize_w 1920 --resize_h 1080 \
  --out clip_1/debug/keyframe_000438_proj_y.png
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


# ----------------------------
# COLMAP utils
# ----------------------------
def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP qvec is [qw, qx, qy, qz]. Returns R (world->cam)."""
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    # normalize (cheap safety)
    n = np.linalg.norm(qvec)
    if n > 0:
        qvec = qvec / n
    qw, qx, qy, qz = qvec

    # standard quaternion-to-rotation
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def read_cameras_simple_radial(cameras_txt: str) -> Tuple[float, float, float, float, float, str]:
    """
    Returns: (f, cx, cy, k1, width, height, model)
    Only supports SIMPLE_RADIAL.
    """
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Bad cameras.txt line: {line}")
            _cam_id = int(parts[0])
            model = parts[1]
            width = float(parts[2])
            height = float(parts[3])
            if model != "SIMPLE_RADIAL":
                raise ValueError(f"Only SIMPLE_RADIAL supported in this script. Got: {model}")
            f_ = float(parts[4])
            cx = float(parts[5])
            cy = float(parts[6])
            k1 = float(parts[7])
            return f_, cx, cy, k1, width, height, model
    raise FileNotFoundError("No valid camera line found in cameras.txt")


def read_images_txt(images_txt: str) -> List[dict]:
    """
    Parses COLMAP images.txt (text export).
    Each image has 2 lines:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] ...
    """
    imgs = []
    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Bad images.txt line: {line}")

        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)  # qw qx qy qz
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)  # tx ty tz
        cam_id = int(parts[8])
        name = parts[9]

        # skip points2D line
        if i < len(lines):
            i += 1

        imgs.append({
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name,
        })
    return imgs


def sort_key_from_name(name: str) -> Tuple[int, str]:
    """
    Try to sort by frame number in filename; fallback lexicographic.
    Example: images/000438.png -> 438
    """
    base = os.path.basename(name)
    m = re.search(r"(\d+)", base)
    if m:
        return (int(m.group(1)), name)
    return (10**18, name)


# ----------------------------
# Projection
# ----------------------------
def apply_simple_radial_distortion(x: float, y: float, k1: float) -> Tuple[float, float]:
    r2 = x*x + y*y
    s = 1.0 + k1 * r2
    return x * s, y * s


def project_points(
    fx: float, fy: float, cx: float, cy: float,
    Rcw: np.ndarray, tcw: np.ndarray,
    Pw_list: List[np.ndarray],
    w: int, h: int,
    k1: float,
    min_z: float = 0.2
) -> List[Tuple[float, float]]:
    """
    World -> Camera: Pc = Rcw * Pw + tcw
    Pin-hole + SIMPLE_RADIAL distortion.
    Returns list of (u,v) in *same pixel coordinates as K*.
    """
    out = []
    Rcw = np.asarray(Rcw, dtype=np.float64).reshape(3, 3)
    tcw = np.asarray(tcw, dtype=np.float64).reshape(3)

    for Pw in Pw_list:
        Pw = np.asarray(Pw, dtype=np.float64).reshape(3)
        Pc = Rcw @ Pw + tcw  # camera coords
        z = Pc[2]
        if z <= min_z:
            continue

        x = Pc[0] / z
        y = Pc[1] / z

        x, y = apply_simple_radial_distortion(x, y, k1)

        u = fx * x + cx
        v = fy * y + cy

        if 0 <= u < w and 0 <= v < h:
            out.append((u, v))
    return out


def forward_axis_to_vec(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if axis == "neg_x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "neg_y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if axis == "neg_z":
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    raise ValueError(f"Unknown forward_axis: {axis}. Use x/y/z/neg_x/neg_y/neg_z.")


# ----------------------------
# GT parsing (robust)
# ----------------------------
def extract_clip_entry(gt_obj, clip_id: str):
    if gt_obj is None:
        return None
    if isinstance(gt_obj, list):
        for item in gt_obj:
            if isinstance(item, dict) and item.get("clip_id") == clip_id:
                return item
        return None
    if isinstance(gt_obj, dict):
        # sometimes dict keyed by clip_id
        if clip_id in gt_obj and isinstance(gt_obj[clip_id], dict):
            return gt_obj[clip_id]
        # or nested under "data"
        if "data" in gt_obj and isinstance(gt_obj["data"], dict) and clip_id in gt_obj["data"]:
            return gt_obj["data"][clip_id]
    return None


def extract_gt_traj_from_clip_entry(clip_entry) -> Optional[list]:
    if not isinstance(clip_entry, dict):
        return None

    # common: clip_entry["texts"] is list of dicts, one contains "gt_traj"
    texts = clip_entry.get("texts", None)
    if isinstance(texts, list):
        for t in texts:
            if isinstance(t, dict) and "gt_traj" in t:
                return t["gt_traj"]
    if isinstance(texts, dict):
        if "gt_traj" in texts:
            return texts["gt_traj"]

    # fallback: direct key
    if "gt_traj" in clip_entry:
        return clip_entry["gt_traj"]

    return None



# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras_txt", required=True)
    ap.add_argument("--images_txt", required=True)
    ap.add_argument("--image_root", required=True, help="Root folder that contains the clip images (e.g., ./dataset/clip_1)")
    ap.add_argument("--keyframe", required=True, help="Path relative to image_root (e.g., images/000438.png) matching COLMAP NAME")
    ap.add_argument("--gt_json", default=None)
    ap.add_argument("--clip_id", default=None)
    ap.add_argument("--lookahead_d", type=float, default=2.0)
    ap.add_argument("--forward_axis", type=str, default="z", help="x/y/z/neg_x/neg_y/neg_z (in camera coords)")
    ap.add_argument("--resize_w", type=int, default=None)
    ap.add_argument("--resize_h", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Camera intrinsics
    f, cx, cy, k1, cam_w, cam_h, model = read_cameras_simple_radial(args.cameras_txt)
    fx = fy = f

    # Images/poses
    imgs = read_images_txt(args.images_txt)
    imgs.sort(key=lambda it: sort_key_from_name(it["name"]))

    # Find keyframe item
    key_item = None
    for it in imgs:
        if it["name"] == args.keyframe:
            key_item = it
            break
    if key_item is None:
        # fallback match by basename
        kb = os.path.basename(args.keyframe)
        for it in imgs:
            if os.path.basename(it["name"]) == kb:
                key_item = it
                break
    if key_item is None:
        raise ValueError(f"Keyframe not found in images.txt: {args.keyframe}")

    key_idx = next(i for i, it in enumerate(imgs) if it["name"] == key_item["name"])

    # Load GT (optional) to decide N
    gt_traj = None
    if args.gt_json and args.clip_id:
        with open(os.path.expanduser(args.gt_json), "r") as fgt:
            gt_obj = json.load(fgt)
        clip_entry = extract_clip_entry(gt_obj, args.clip_id)
        gt_traj = extract_gt_traj_from_clip_entry(clip_entry)

    N = len(gt_traj) if (gt_traj is not None and isinstance(gt_traj, list) and len(gt_traj) > 0) else 12

    # Read image
    img_path = os.path.join(args.image_root, key_item["name"])
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h0, w0 = img.shape[:2]
    sx = sy = 1.0
    if args.resize_w is not None and args.resize_h is not None:
        sx = args.resize_w / float(w0)
        sy = args.resize_h / float(h0)
        img = cv2.resize(img, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]

    # ✅ Scale intrinsics to resized image coordinates
    fx_s = fx * sx
    fy_s = fy * sy
    cx_s = cx * sx
    cy_s = cy * sy

    # Keyframe pose (world->cam)
    R_ref = qvec2rotmat(key_item["qvec"])
    t_ref = key_item["tvec"]

    # future-only lookahead_world (N poses from key_idx)
    # future-only indices
    end_idx = min(key_idx + N, len(imgs))

    # 先把 future-only 的 camera centers 算出來
    Cs = []
    for it in imgs[key_idx:end_idx]:
        R = qvec2rotmat(it["qvec"])
        t = it["tvec"]
        C = -R.T @ t
        Cs.append(C)

    lookahead_world = []
    for i in range(len(Cs)):
        # motion direction in world
        if i < len(Cs) - 1:
            dir_w = Cs[i + 1] - Cs[i]
        else:
            dir_w = Cs[i] - Cs[i - 1] if i > 0 else None

        ok = False
        if dir_w is not None:
            n = np.linalg.norm(dir_w)
            if n > 1e-9:
                dir_w = dir_w / n
                P = Cs[i] + args.lookahead_d * dir_w
                lookahead_world.append(P)
                ok = True

        # fallback: 用你指定的 forward_axis（避免 motion direction 退化）
        if not ok:
            forward_cam = forward_axis_to_vec(args.forward_axis)
            it = imgs[key_idx + i]
            R = qvec2rotmat(it["qvec"])
            t = it["tvec"]
            C = -R.T @ t
            forward_world = R.T @ forward_cam
            P = C + args.lookahead_d * forward_world
            lookahead_world.append(P)
    print(f"[INFO] lookahead_world count = {len(lookahead_world)} (N={N})")
    # Project pred points
    uv_pred = project_points(
        fx_s, fy_s, cx_s, cy_s,
        R_ref, t_ref,
        lookahead_world,
        w, h,
        k1=k1,
        min_z=0.2
    )

    pred_traj = []
    for (u, v) in uv_pred:
        uu = int(round(u))
        vv = int(round(v))
        pred_traj.append([uu, vv])
        cv2.circle(img, (uu, vv), 4, (0, 255, 0), -1)

    # Project / draw GT (optional)
    uv_gt = None
    if gt_traj is not None and isinstance(gt_traj, list) and len(gt_traj) > 0:
        # detect dim
        dim = len(gt_traj[0]) if isinstance(gt_traj[0], (list, tuple)) else None
        if dim == 2:
            # GT already in original pixel coords -> scale to resized
            uv_gt = [(p[0] * sx, p[1] * sy) for p in gt_traj]
        elif dim == 3:
            Pw_gt = [np.array(p, dtype=np.float64) for p in gt_traj]
            uv_gt = project_points(
                fx_s, fy_s, cx_s, cy_s,
                R_ref, t_ref,
                Pw_gt,
                w, h,
                k1=k1,
                min_z=0.2
            )
        else:
            print("[WARN] Unknown GT traj format; skip drawing GT.")

    if uv_gt is not None:
        gt_traj_draw = []
        for (u, v) in uv_gt:
            uu = int(round(u))
            vv = int(round(v))
            gt_traj_draw.append([uu, vv])
            if 0 <= uu < w and 0 <= vv < h:
                cv2.circle(img, (uu, vv), 4, (0, 0, 255), -1)

        # aligned print
        L = min(len(pred_traj), len(gt_traj_draw))
        print(pred_traj[:L])
        print("=" * 10)
        print(gt_traj_draw[:L])
    else:
        print(pred_traj)

    # Save output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
