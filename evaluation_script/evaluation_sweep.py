import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

# text metrics
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False


# -----------------------------
# Parsing utilities
# -----------------------------

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def extract_waypoint_pairs_from_text(text: str) -> List[List[float]]:
    """
    Robust parser:
    - Accepts "(x, y)" pairs
    - Ignores extra values e.g. "(x, y, w)" or "(x, y, w, h)" by taking first two numbers
    - Works even if text lacks 'Waypoints:' prefix
    """
    if not text:
        return []

    # Find all (...) chunks
    tuples = re.findall(r"\(([^()]*)\)", text)
    points = []
    for t in tuples:
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        if len(nums) >= 2:
            try:
                x = float(nums[0])
                y = float(nums[1])
                points.append([x, y])
            except Exception:
                continue
    return points


def extract_waypoints_any(sample: Dict[str, Any], pred: bool = True) -> List[List[float]]:
    key = "predicted_waypoints" if pred else "groundtruth_waypoints"
    ans_key = "predicted_answer" if pred else "groundtruth_answer"

    # 1) direct field
    wps = sample.get(key, None)
    if isinstance(wps, list) and len(wps) > 0:
        out = []
        for p in wps:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    out.append([float(p[0]), float(p[1])])
                except Exception:
                    pass
        if out:
            return out

    # 2) parse from answer text
    text = safe_str(sample.get(ans_key, ""))
    return extract_waypoint_pairs_from_text(text)


def extract_reasoning_text(answer: str) -> str:
    """
    Try to parse reasoning if present; otherwise return the whole text.
    """
    if not answer:
        return ""

    # Preferred explicit format
    m = re.search(r"Reasoning:\s*([\s\S]*)", answer, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # If "Waypoints:" exists, remove it and nearby list to isolate reasoning
    if "Waypoints:" in answer:
        text = re.sub(r"Waypoints:\s*\[[\s\S]*?\]", "", answer, flags=re.IGNORECASE)
        return text.strip()

    return answer.strip()


# -----------------------------
# Waypoint metrics
# -----------------------------

def euclidean(p: List[float], q: List[float]) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def waypoint_l2_distance_error(
    pred_wps: List[List[float]],
    gt_wps: List[List[float]],
    mode: str = "mean"
) -> Optional[float]:
    """
    Compare aligned points up to min length.
    mode:
      - mean: average L2 across aligned waypoints
      - max: max L2 across aligned waypoints (similar to some papers' per-sample max)
    """
    if not pred_wps or not gt_wps:
        return None
    n = min(len(pred_wps), len(gt_wps))
    if n <= 0:
        return None

    dists = [euclidean(pred_wps[i], gt_wps[i]) for i in range(n)]
    if mode == "max":
        return max(dists)
    return sum(dists) / len(dists)


# -----------------------------
# Text metrics
# -----------------------------


# -----------------------------
# Resolution sweep / alignment helpers
# -----------------------------

def parse_wh_list(spec: Optional[str]) -> List[Tuple[int, int]]:
    """Parse resolutions like '1920x1080,1280x720,640x360'"""
    if not spec:
        return []
    out: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad resolution token: {part}")
        w, h = part.split("x", 1)
        out.append((int(w), int(h)))
    return out


def get_image_wh(image_path: str) -> Tuple[int, int]:
    """Return (W,H) of image."""
    if not image_path:
        raise ValueError("Missing image path for resolution sweep.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Pillow first
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            W, H = im.size
        return int(W), int(H)
    except Exception:
        pass

    # OpenCV fallback
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread failed")
        H, W = img.shape[:2]
        return int(W), int(H)
    except Exception as e:
        raise RuntimeError(f"Cannot read image size for: {image_path} ({e})")


def resize_waypoints(wps: List[List[float]], from_wh: Tuple[int, int], to_wh: Tuple[int, int]) -> List[List[float]]:
    """Scale (x,y) from (W,H) -> (W2,H2)."""
    if not wps:
        return []
    W, H = from_wh
    W2, H2 = to_wh
    sx = W2 / float(W)
    sy = H2 / float(H)
    return [[float(p[0]) * sx, float(p[1]) * sy] for p in wps]


def align_waypoints(pred_wps: List[List[float]], gt_wps: List[List[float]], align_mode: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    align_mode:
      - 'min': truncate both to min length (original behavior)
      - 'gt' : truncate pred to len(gt); if pred shorter, use pred length
    """
    if not pred_wps or not gt_wps:
        return pred_wps, gt_wps

    if align_mode == "gt":
        K = len(gt_wps)
        pred2 = pred_wps[:K]
        gt2 = gt_wps[:len(pred2)]
        return pred2, gt2

    n = min(len(pred_wps), len(gt_wps))
    return pred_wps[:n], gt_wps[:n]
def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def lcs_length(a: List[str], b: List[str]) -> int:
    # ROUGE-L fallback if rouge_score package is unavailable
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_f1_fallback(pred: str, ref: str) -> float:
    p_toks = tokenize_simple(pred)
    r_toks = tokenize_simple(ref)
    if not p_toks or not r_toks:
        return 0.0
    lcs = lcs_length(p_toks, r_toks)
    prec = lcs / len(p_toks) if p_toks else 0.0
    rec = lcs / len(r_toks) if r_toks else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_text_metrics(pred: str, ref: str) -> Dict[str, Optional[float]]:
    pred = safe_str(pred)
    ref = safe_str(ref)

    out = {
        "bleu4": None,
        "rougeL_f1": None,
        "meteor": None,
    }

    # BLEU-4
    if _HAS_NLTK:
        chencherry = SmoothingFunction()
        ref_tokens = tokenize_simple(ref)
        pred_tokens = tokenize_simple(pred)
        if ref_tokens and pred_tokens:
            out["bleu4"] = float(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=chencherry.method1
                )
            )

    # ROUGE-L
    if _HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(ref, pred)["rougeL"]
        out["rougeL_f1"] = float(score.fmeasure)
    else:
        out["rougeL_f1"] = float(rouge_l_f1_fallback(pred, ref))

    # METEOR
    if _HAS_NLTK:
        try:
            out["meteor"] = float(meteor_score([tokenize_simple(ref)], tokenize_simple(pred)))
        except Exception:
            # fallback to None if punkt/wordnet resources missing
            out["meteor"] = None

    return out


# -----------------------------
# GPT-4o Grader (optional)
# -----------------------------

GPT_RUBRIC_TEMPLATE = """Please evaluate the predicted answer on a scale from 0 to 100, using the following rubric focused on socially compliant driving behavior. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity.

1. Action Appropriateness (15 points): Does the predicted driving action (e.g., turn left, stop) align with the intended high-level behavior required in the scenario?
2. Motion Appropriateness (15 points): Does the predicted motion (e.g., decelerate, maintain speed) reflect a safe and context-appropriate response to dynamic elements in the scene (e.g., pedestrians)?
3. Social Cue Understanding (20 points): Does the model correctly interpret social cues such as "waiting to cross," "talking in group," or "getting into a vehicle"?
4. Agent Importance Awareness (15 points): Does the answer demonstrate awareness of which agents are most likely to affect ego-vehicle behavior?
5. Driving Context Appropriateness (15 points): Is the predicted action logical given the broader traffic context (e.g., narrow road, parked vehicles, group behavior)?
6. Conciseness and Clarity (10 points): Is the answer clearly written and easy to understand without redundancy?
7. Grammar and Structure (10 points): Is the response grammatically correct and well-structured?

Please return your evaluation in JSON with this exact schema:
{{
  "total_score": <0-100 number>,
  "subscores": {{
    "action_appropriateness": <0-15>,
    "motion_appropriateness": <0-15>,
    "social_cue_understanding": <0-20>,
    "agent_importance_awareness": <0-15>,
    "driving_context_appropriateness": <0-15>,
    "conciseness_and_clarity": <0-10>,
    "grammar_and_structure": <0-10>
  }},
  "summary": "<short summary>"
}}

Here is the predicted answer:
{predicted_answer}

Here is the groundtruth answer:
{groundtruth_answer}
"""


def grade_with_openai_gpt4o(
    predicted_answer: str,
    groundtruth_answer: str,
    model_name: str = "gpt-4o"
) -> Optional[Dict[str, Any]]:
    """
    Optional grader. Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI
    except Exception:
        return None

    api_key = os.environ.get("OPENAI_API_KEY", None)
    if not api_key:
        return None

    prompt = GPT_RUBRIC_TEMPLATE.format(
        predicted_answer=predicted_answer,
        groundtruth_answer=groundtruth_answer
    )

    client = OpenAI(api_key=api_key)

    try:
        # Using Responses API style
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0.0,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # fallback parse
            text = str(resp)

        # Try JSON parse directly; if model wrapped text, extract JSON block
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"raw_text": text}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Core evaluation
# -----------------------------

def evaluate_one_sample(
    sample: Dict[str, Any],
    l2_mode: str = "mean",
    use_reasoning_only_for_text_metrics: bool = False,
    use_gpt4o_grader: bool = False,
    gpt4o_model: str = "gpt-4o",
    align_mode: str = "min",
    resolution_sweep: Optional[List[Tuple[int, int]]] = None,
    image_root: str = "",
) -> Dict[str, Any]:
    pred_answer = safe_str(sample.get("predicted_answer", ""))
    gt_answer = safe_str(sample.get("groundtruth_answer", ""))

    pred_wps = extract_waypoints_any(sample, pred=True)
    gt_wps = extract_waypoints_any(sample, pred=False)

    # text metric target
    if use_reasoning_only_for_text_metrics:
        pred_text = extract_reasoning_text(pred_answer)
        gt_text = extract_reasoning_text(gt_answer)
    else:
        pred_text = pred_answer
        gt_text = gt_answer

    text_metrics = compute_text_metrics(pred_text, gt_text)

    # Align before computing base L2
    pred_aligned, gt_aligned = align_waypoints(pred_wps, gt_wps, align_mode)
    l2_err = waypoint_l2_distance_error(pred_aligned, gt_aligned, mode=l2_mode)

    # Resolution sweep: scale BOTH pred & gt into target resolutions, then compute L2
    sweep = None
    sweep_from_wh = None
    if resolution_sweep:
        img_path = safe_str(sample.get("image", ""))
        if image_root and img_path and (not os.path.isabs(img_path)):
            img_path = os.path.join(image_root, img_path)
        W, H = get_image_wh(img_path)
        sweep_from_wh = [W, H]
        sweep = {}
        for (W2, H2) in resolution_sweep:
            pred_rs = resize_waypoints(pred_wps, (W, H), (W2, H2))
            gt_rs = resize_waypoints(gt_wps, (W, H), (W2, H2))
            pred_rs, gt_rs = align_waypoints(pred_rs, gt_rs, align_mode)
            sweep[f"{W2}x{H2}"] = waypoint_l2_distance_error(pred_rs, gt_rs, mode=l2_mode)

    result = {
        "sample_id": sample.get("sample_id", sample.get("clip_id", sample.get("id", None))),
        "mode": sample.get("mode", None),
        "text_metrics": text_metrics,
        "waypoint_metrics": {
            "pred_waypoint_count": len(pred_wps),
            "gt_waypoint_count": len(gt_wps),
            "l2_distance_error": l2_err,
            "l2_mode": l2_mode,
            "align_mode": align_mode,
            "l2_distance_error_sweep": sweep,
            "sweep_from_wh": sweep_from_wh,
        },
        "parsed": {
            "predicted_waypoints": pred_wps,
            "groundtruth_waypoints": gt_wps,
            "pred_reasoning": extract_reasoning_text(pred_answer),
            "gt_reasoning": extract_reasoning_text(gt_answer),
        }
    }

    if use_gpt4o_grader:
        grader = grade_with_openai_gpt4o(
            predicted_answer=pred_answer,
            groundtruth_answer=gt_answer,
            model_name=gpt4o_model,
        )
        result["gpt4o_reasoning_grader"] = grader

    return result


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean_ignore_none(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    bleu4 = []
    rougeL = []
    meteor = []
    l2s = []
    pred_counts = []
    gt_counts = []
    gpt_scores = []
    sweep_bucket: Dict[str, List[float]] = {}

    for r in results:
        tm = r.get("text_metrics", {})
        wm = r.get("waypoint_metrics", {})
        bleu4.append(tm.get("bleu4"))
        rougeL.append(tm.get("rougeL_f1"))
        meteor.append(tm.get("meteor"))
        l2s.append(wm.get("l2_distance_error"))
        pred_counts.append(wm.get("pred_waypoint_count"))
        gt_counts.append(wm.get("gt_waypoint_count"))

        sw = wm.get("l2_distance_error_sweep", None)
        if isinstance(sw, dict):
            for k, v in sw.items():
                if isinstance(v, (int, float)):
                    sweep_bucket.setdefault(k, []).append(float(v))

        g = r.get("gpt4o_reasoning_grader", None)
        if isinstance(g, dict):
            if isinstance(g.get("total_score"), (int, float)):
                gpt_scores.append(float(g["total_score"]))

    sweep_avg = {k: (sum(vs) / len(vs)) for k, vs in sweep_bucket.items()} if sweep_bucket else None

    # parse success: at least 1 predicted waypoint
    parse_success = sum(1 for c in pred_counts if isinstance(c, int) and c > 0)
    total = len(results)

    return {
        "num_samples": total,
        "text_metrics_avg": {
            "bleu4": mean_ignore_none(bleu4),
            "rougeL_f1": mean_ignore_none(rougeL),
            "meteor": mean_ignore_none(meteor),
        },
        "waypoint_metrics_avg": {
            "l2_distance_error": mean_ignore_none(l2s),
            "pred_waypoint_count": mean_ignore_none(pred_counts),
            "gt_waypoint_count": mean_ignore_none(gt_counts),
            "waypoint_parse_success_rate": (parse_success / total) if total > 0 else None,
            "l2_distance_error_sweep_avg": sweep_avg,
        },
        "gpt4o_reasoning_grader_avg": {
            "total_score": mean_ignore_none(gpt_scores) if gpt_scores else None
        }
    }


# -----------------------------
# IO helpers
# -----------------------------

def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # support {"records":[...]} or single record dict
        if "records" in data and isinstance(data["records"], list):
            return data["records"]
        return [data]
    raise ValueError("Unsupported JSON format.")


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True, help="Inference outputs JSON (single record or list).")
    ap.add_argument("--output_json", type=str, required=True, help="Where to save detailed eval results.")
    ap.add_argument("--summary_json", type=str, default=None, help="Optional summary JSON path.")
    ap.add_argument("--l2_mode", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--reasoning_only_text_metrics", action="store_true")
    ap.add_argument("--use_gpt4o_grader", action="store_true")
    ap.add_argument("--gpt4o_model", type=str, default="gpt-4o")
    ap.add_argument("--align_mode", type=str, default="min", choices=["min", "gt"],
                    help="Waypoint alignment: min length or truncate pred to gt length.")
    ap.add_argument("--resolution_sweep", type=str, default=None,
                    help='Comma-separated resolutions, e.g. "1920x1080,1280x720,640x360"')
    ap.add_argument("--image_root", type=str, default="",
                    help="Optional prefix for sample['image'] if paths are relative.")
    args = ap.parse_args()

    records = load_json_records(args.input_json)
    results = []

    for i, rec in enumerate(records):
        try:
            sweep_list = parse_wh_list(args.resolution_sweep) if args.resolution_sweep else None
            r = evaluate_one_sample(
                rec,
                l2_mode=args.l2_mode,
                use_reasoning_only_for_text_metrics=args.reasoning_only_text_metrics,
                use_gpt4o_grader=args.use_gpt4o_grader,
                gpt4o_model=args.gpt4o_model,
                align_mode=args.align_mode,
                resolution_sweep=sweep_list,
                image_root=args.image_root,
            )
            results.append(r)
        except Exception as e:
            results.append({
                "sample_id": rec.get("sample_id", rec.get("clip_id", i)) if isinstance(rec, dict) else i,
                "error": str(e)
            })

    summary = aggregate_results([r for r in results if "error" not in r])

    payload = {
        "summary": summary,
        "results": results,
    }

    save_json(args.output_json, payload)
    if args.summary_json:
        save_json(args.summary_json, summary)

    print("==== Evaluation Summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved detailed results to: {args.output_json}")
    if args.summary_json:
        print(f"Saved summary to: {args.summary_json}")


if __name__ == "__main__":
    main()
