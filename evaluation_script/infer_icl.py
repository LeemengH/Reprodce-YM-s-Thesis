# infer_icl.py
# Supports:
# - Base model inference with In-Context Learning (ICL)
# - Multi-image ICL (default) or Text-only ICL (--text_only_icl)
# - No LoRA weights required

import os
import re
import json
import argparse
import random
from typing import Optional, Any, Dict, List

import torch
from PIL import Image

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
)

SYSTEM_PROMPT = (
    "You are a helpful and socially-aware planning module for understanding "
    "urban driving scenes, social cues, and outputting future waypoints "
    "for the ego vehicle."
)


def build_basic_user_prompt(social_text: str) -> str:
    return (
        "What should the ego vehicle do next? Reasoning the plan. "
        "And plan a pixel-wise future trajectory in the format [(x1, y1), (x2, y2), ...]. "
        "Social cue: " + (social_text or "")
    )


def resolve_image_path(image_root: str, p: Optional[str]) -> str:
    if not p:
        raise ValueError("Image path is empty.")
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(image_root, p)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_key(x: Dict[str, Any]):
    return x.get("clip_id") or x.get("id") or x.get("image") or x.get("img") or x.get("image_path")


def _extract_text_entry(sample_obj: Dict[str, Any]) -> Dict[str, Any]:
    texts = sample_obj.get("texts")
    if isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], dict):
        return texts[0]
    return sample_obj


def load_all_paired_samples(cue_json_path: str, nocue_json_path: str) -> List[Dict[str, Any]]:
    cue_data = _read_json(cue_json_path)
    nocue_data = _read_json(nocue_json_path)

    if not isinstance(cue_data, list) or not isinstance(nocue_data, list):
        raise ValueError("Expected both cue and no_cue json files to be lists.")

    cue_map = {_get_key(x): x for x in cue_data if _get_key(x) is not None}
    nocue_map = {_get_key(x): x for x in nocue_data if _get_key(x) is not None}
    common = sorted(set(cue_map.keys()) & set(nocue_map.keys()))

    paired = []
    for key in common:
        c = cue_map[key]
        n = nocue_map[key]

        ct = _extract_text_entry(c)
        nt = _extract_text_entry(n)

        image_path = (
            c.get("image")
            or c.get("img")
            or c.get("image_path")
            or ct.get("image")
            or ct.get("image_path")
        )

        paired.append({
            "key": key,
            "image_path": image_path,
            "cue_text": ct.get("social_cue", "") or ct.get("cue", "") or "",
            "nocue_text": nt.get("social_cue", "") or nt.get("cue", "") or "",
            "gt_traj": ct.get("gt_traj"),
            "reasoning_gt": (
                ct.get("assistant")
                or ct.get("answer")
                or ct.get("output")
                or ""
            ),
            "raw_cue": c,
            "raw_nocue": n,
        })

    if len(paired) == 0:
        raise ValueError("No paired samples found between cue and no_cue json files.")
    return paired


def load_sample_from_json(cue_json_path: str, nocue_json_path: str, index: int = 0) -> Dict[str, Any]:
    paired = load_all_paired_samples(cue_json_path, nocue_json_path)
    return paired[index % len(paired)]


def sample_icl_examples(all_pairs: List[Dict[str, Any]], query_key: Any, k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    pool = []
    for x in all_pairs:
        if str(x.get("key")) == str(query_key):
            continue
        gt_traj = x.get("gt_traj")
        if not isinstance(gt_traj, list) or len(gt_traj) == 0:
            continue
        pool.append(x)

    if len(pool) == 0:
        return []

    rnd = random.Random(seed)
    if len(pool) <= k:
        rnd.shuffle(pool)
        return pool
    return rnd.sample(pool, k)


def format_waypoints_for_demo(gt_traj) -> str:
    pairs = []
    if isinstance(gt_traj, list):
        for p in gt_traj:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pairs.append(f"({int(p[0])}, {int(p[1])})")
                except Exception:
                    continue
    return "[" + ", ".join(pairs) + "]"


def build_output_from_example(example: Dict[str, Any]) -> str:
    wp_str = format_waypoints_for_demo(example.get("gt_traj"))
    reasoning = (example.get("reasoning_gt") or "").strip()
    return f"Waypoints: {wp_str}\nReasoning: {reasoning}"


def parse_waypoints_from_text(output_text: str):
    if not output_text:
        return None
    tuples = re.findall(r"\(([^()]*)\)", output_text)
    out = []
    for t in tuples:
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        if len(nums) >= 2:
            try:
                out.append([float(nums[0]), float(nums[1])])
            except Exception:
                continue
    return out if out else None


def parse_waypoints_tagged(output_text: str):
    if not output_text:
        return None
    m = re.search(r"Waypoints:\s*(\[[\s\S]*?\])", output_text, flags=re.IGNORECASE)
    if m:
        return parse_waypoints_from_text(m.group(1))
    return parse_waypoints_from_text(output_text)


def parse_reasoning_tagged(output_text: str):
    if not output_text:
        return None
    m = re.search(r"Reasoning:\s*([\s\S]*)", output_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


@torch.no_grad()
def generate_one(
    model,
    processor,
    images: List[Image.Image],
    prompt_text: str,
    device: str,
    max_new_tokens: int = 256,
):
    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=4096,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0][prompt_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="In-Context Learning Inference for LLaVA (No LoRA)")

    # model
    ap.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")

    # data / input
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--image_path", type=str, default=None)
    ap.add_argument("--social_cue_text", type=str, default=None)

    # paired json mode
    ap.add_argument("--cue_json", type=str, default=None)
    ap.add_argument("--nocue_json", type=str, default=None)
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--use_masked_prompt", action="store_true", help="Use no_cue text for query prompt")

    # ICL settings
    ap.add_argument("--icl_k", type=int, default=1, help="Number of ICL examples (default: 1)")
    ap.add_argument("--icl_seed", type=int, default=42)
    ap.add_argument("--icl_from", type=str, default="cue", choices=["cue", "nocue"], help="Which social cue variant to use in ICL examples")
    ap.add_argument("--text_only_icl", action="store_true", help="If set, ICL examples will not include images (text-only prompts).")

    # generation / runtime
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 (prefer bf16 on modern GPUs)")
    ap.add_argument("--save_json", type=str, default=None, help="Save one inference record to JSON")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[INFO] device={device}, dtype={dtype}")

    # Build processor manually for compatibility
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[INFO] Loading base model for ICL...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()

    sample = None
    all_pairs = None

    # Resolve query sample
    if args.cue_json and args.nocue_json:
        # Check if enough samples for ICL
        all_pairs = load_all_paired_samples(args.cue_json, args.nocue_json)
        sample = all_pairs[args.sample_index % len(all_pairs)]
        query_image_path = resolve_image_path(args.image_root, sample["image_path"])
        query_social_text = sample["nocue_text"] if args.use_masked_prompt else sample["cue_text"]

        print(f"[INFO] sample key: {sample.get('key')}")
        print(f"[INFO] query image path: {query_image_path}")
        gt_traj = sample.get("gt_traj")
        if isinstance(gt_traj, list):
            print(f"[INFO] gt_traj length: {len(gt_traj)}")
    else:
        if args.image_path is None or args.social_cue_text is None:
            raise ValueError("Provide either (--cue_json and --nocue_json) OR (--image_path and --social_cue_text).")
        query_image_path = resolve_image_path(args.image_root, args.image_path)
        query_social_text = args.social_cue_text

    # Sample ICL examples
    icl_examples = []
    if args.icl_k > 0:
        if not (args.cue_json and args.nocue_json):
            print("[WARN] --icl_k > 0 but no paired json provided. ICL disabled.")
        else:
            icl_examples = sample_icl_examples(
                all_pairs=all_pairs,
                query_key=sample.get("key") if sample else None,
                k=args.icl_k,
                seed=args.icl_seed,
            )
            print(f"[INFO] ICL examples sampled: {len(icl_examples)} (icl_from={args.icl_from})")

    # Build prompt and images list
    images = []
    
    if args.text_only_icl:
        # Legacy text-only ICL format
        lines = []
        if icl_examples:
            lines.append("Here are example input-output pairs showing the expected format and behavior.\n")
            for i, ex in enumerate(icl_examples, 1):
                social_text = ex.get("cue_text", "") if args.icl_from == "cue" else ex.get("nocue_text", "")
                user_part = build_basic_user_prompt(social_text)
                assistant_part = build_output_from_example(ex)
                lines.append(f"[Example {i}]")
                lines.append(f"User: {user_part}")
                lines.append(f"Assistant: {assistant_part}\n")
        
        icl_block = "\n".join(lines)
        query_user_prompt = build_basic_user_prompt(query_social_text)
        
        if icl_block:
            full_user_prompt = icl_block + "[Now solve the following case]\n" + query_user_prompt
        else:
            full_user_prompt = query_user_prompt

        prompt_text = f"{SYSTEM_PROMPT}\nUSER: <image>\n{full_user_prompt}\nASSISTANT:"
        
        # Only 1 image
        query_img = Image.open(query_image_path).convert("RGB")
        images.append(query_img)
        
    else:
        # Multi-image True ICL format
        prompt_text = SYSTEM_PROMPT + "\n"
        
        for ex in icl_examples:
            ex_img_path = resolve_image_path(args.image_root, ex["image_path"])
            ex_img = Image.open(ex_img_path).convert("RGB")
            images.append(ex_img)
            
            social_text = ex.get("cue_text", "") if args.icl_from == "cue" else ex.get("nocue_text", "")
            user_part = build_basic_user_prompt(social_text)
            assistant_part = build_output_from_example(ex)
            
            prompt_text += f"USER: <image>\n{user_part}\nASSISTANT: {assistant_part}</s>"
        
        # Add query
        query_img = Image.open(query_image_path).convert("RGB")
        images.append(query_img)
        query_user_prompt = build_basic_user_prompt(query_social_text)
        prompt_text += f"USER: <image>\n{query_user_prompt}\nASSISTANT:"

    print("\n[INFO] Constructed Prompt:\n" + "-"*40)
    print(prompt_text)
    print("-" * 40 + "\n")
    print(f"[INFO] Total images in prompt: {len(images)}")
    
    print("\n[INFO] Generating...\n")
    output_text = generate_one(
        model=model,
        processor=processor,
        images=images,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    parsed_wps = parse_waypoints_tagged(output_text)
    parsed_reasoning = parse_reasoning_tagged(output_text)

    print("===== MODEL OUTPUT =====\n")
    print(output_text)

    print("\n===== PARSED =====\n")
    print("Waypoints:", parsed_wps)
    print("Reasoning:", parsed_reasoning)

    # Save JSON record (single sample)
    if args.save_json is not None:
        gt_answer = None
        gt_waypoints = None
        sample_id = None
        mode = "masked" if args.use_masked_prompt else "cue"

        if sample is not None:
            sample_id = str(sample.get("key"))
            gt_waypoints = sample.get("gt_traj")
            gt_reasoning = sample.get("reasoning_gt", "")
            gt_answer = f"Waypoints: {format_waypoints_for_demo(gt_waypoints)}\nReasoning: {gt_reasoning}"

        record = {
            "sample_id": sample_id,
            "image": query_image_path,
            "mode": mode,
            "predicted_answer": output_text,
            "groundtruth_answer": gt_answer,
            "predicted_waypoints": parsed_wps,
            "groundtruth_waypoints": gt_waypoints,
            "meta": {
                "base_model": args.base_model,
                "max_new_tokens": args.max_new_tokens,
                "use_masked_prompt": args.use_masked_prompt,
                "icl_k": args.icl_k,
                "icl_seed": args.icl_seed,
                "icl_from": args.icl_from,
                "text_only_icl": args.text_only_icl,
            }
        }

        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Saved inference record to: {args.save_json}")


if __name__ == "__main__":
    main()
