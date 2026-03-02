import json
import os
import argparse
import cv2
import glob

def get_image_path(clip_id, base_dir=".."):
    """
    Finds the image path for a given clip_id.
    Assumes structure: dataset/images_anonymized/{clip_id}/images/*.png
    """
    pattern = os.path.join(base_dir, "dataset", "images_anonymized", clip_id, "images", "*.png")
    files = glob.glob(pattern)
    if not files:
        return None
    return files[0]

def draw_waypoints(image, waypoints, color, label):
    """
    Draws a sequence of waypoints on the image.
    """
    for i, pt in enumerate(waypoints):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), 5, color, -1)
        if i > 0:
            prev_x, prev_y = int(waypoints[i-1][0]), int(waypoints[i-1][1])
            cv2.line(image, (prev_x, prev_y), (x, y), color, 2)
    
    # Add legend-like text if waypoints exist
    if waypoints:
        first_pt = waypoints[0]
        cv2.putText(image, label, (int(first_pt[0]), int(first_pt[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    parser = argparse.ArgumentParser(description="Visualize waypoints on images from evaluation JSON.")
    parser.add_argument("--input_json", required=True, help="Path to the evaluation JSON file.")
    parser.add_argument("--clip_id", required=True, help="The clip_id to visualize.")
    parser.add_argument("--output_dir", default=".", help="Directory to save the visualization.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: JSON file not found: {args.input_json}")
        return

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    target_sample = None
    for r in results:
        if r.get("sample_id") == args.clip_id:
            target_sample = r
            break
            
    if not target_sample:
        print(f"Error: clip_id '{args.clip_id}' not found in JSON.")
        return
    
    # Extract waypoints
    parsed = target_sample.get("parsed", {})
    gt_waypoints = parsed.get("groundtruth_waypoints", [])
    pred_waypoints = parsed.get("predicted_waypoints", [])
    
    # Resolve image path
    img_path = get_image_path(args.clip_id)
    if not img_path:
        print(f"Error: Could not find image for clip {args.clip_id}")
        return
        
    print(f"Loading image: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    # Draw waypoints
    # GT: Green (0, 255, 0)
    # Pred: Red (0, 0, 255)
    draw_waypoints(image, gt_waypoints, (0, 255, 0), "GroundTruth")
    draw_waypoints(image, pred_waypoints, (0, 0, 255), "Predicted")
    
    # Save output
    output_path = os.path.join(args.output_dir, f"{args.clip_id}_viz.png")
    cv2.imwrite(output_path, image)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    main()
