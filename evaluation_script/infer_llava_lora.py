# infer_llava_lora.py
# Supports:
# - LoRA inference (default)
# - Base model only inference (--disable_lora)
# - cue / masked query prompt (--use_masked_prompt)
# - Save single-record JSON (--save_json)
#
# Notes:
# - This version intentionally does NOT include strict_format_prompt (per your request).
# - It uses manual LlavaProcessor construction for compatibility.

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
from peft import PeftModel


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


def build_llava_prompt(system_prompt: str, user_prompt: str) -> str:
    # LLaVA-1.5 style plain prompt
    return (
        f"{system_prompt}\n"
        f"USER: <image>\n{user_prompt}\n"
        f"ASSISTANT:"
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
    # Tries to support multiple dataset layouts:
    # 1) {"texts":[{...}]}
    # 2) flat dict with keys directly
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





def parse_waypoints_from_text(output_text: str):
    """
    Robust parser:
    - captures tuples like (x,y), (x,y,z), (x,y,z,w)
    - takes first 2 numbers as waypoint
    """
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
    # Optional "Waypoints: [...]" parser; falls back to robust tuple parse
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
    image: Image.Image,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 256,
):
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=4096,  # input prompt cap (not generation cap)
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
    ap = argparse.ArgumentParser()

    # model / weights
    ap.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter path (required unless --disable_lora)")
    ap.add_argument("--disable_lora", action="store_true", help="Use base model only, without loading LoRA adapter")

    # data / input
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--image_path", type=str, default=None)
    ap.add_argument("--social_cue_text", type=str, default=None)

    # paired json mode
    ap.add_argument("--cue_json", type=str, default=None)
    ap.add_argument("--nocue_json", type=str, default=None)
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--use_masked_prompt", action="store_true", help="Use no_cue text for query prompt")

    # generation / runtime
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 (prefer bf16 on modern GPUs)")
    ap.add_argument("--save_json", type=str, default=None, help="Save one inference record to JSON")

    args = ap.parse_args()

    if (not args.disable_lora) and (not args.adapter_path):
        raise ValueError("--adapter_path is required unless you pass --disable_lora")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16
        else:
            # default to fp16 on cuda if neither specified
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[INFO] device={device}, dtype={dtype}")

    # Build processor manually for compatibility
    # --------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # --------------------------------------
    # # ++++++++++++++++++++++++++++++++++++++
    # # Correct way: load official processor
    # processor = LlavaProcessor.from_pretrained(args.base_model)

    # if processor.tokenizer.pad_token_id is None:
    #     processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    # # ++++++++++++++++++++++++++++++++++++++

    print("[INFO] Loading base model...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    # ++++++++++++++++++++++++++++++++++++++
    # IMPORTANT: align embedding size with tokenizer
    base_model.resize_token_embeddings(len(processor.tokenizer))

    # ensure pad token id consistent
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    # ++++++++++++++++++++++++++++++++++++++

    if args.disable_lora:
        print("[INFO] disable_lora=True -> using base model only (no LoRA adapter)")
        model = base_model
    else:
        print(f"[INFO] Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    sample = None

    # Resolve query sample
    if args.cue_json and args.nocue_json:
        sample = load_sample_from_json(args.cue_json, args.nocue_json, args.sample_index)
        image_path = resolve_image_path(args.image_root, sample["image_path"])
        query_social_text = sample["nocue_text"] if args.use_masked_prompt else sample["cue_text"]

        print(f"[INFO] sample key: {sample.get('key')}")
        print(f"[INFO] image path: {image_path}")
        gt_traj = sample.get("gt_traj")
        if isinstance(gt_traj, list):
            print(f"[INFO] gt_traj length: {len(gt_traj)}")
    else:
        if args.image_path is None or args.social_cue_text is None:
            raise ValueError("Provide either (--cue_json and --nocue_json) OR (--image_path and --social_cue_text).")
        image_path = resolve_image_path(args.image_root, args.image_path)
        query_social_text = args.social_cue_text

    full_user_prompt = build_basic_user_prompt(query_social_text)

    prompt_text = build_llava_prompt(SYSTEM_PROMPT, full_user_prompt)

    image = Image.open(image_path).convert("RGB")

    print("\n[INFO] Generating...\n")
    output_text = generate_one(
        model=model,
        processor=processor,
        image=image,
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
            "image": image_path,
            "mode": mode,
            "predicted_answer": output_text,
            "groundtruth_answer": gt_answer,
            "predicted_waypoints": parsed_wps,       # store parsed result too
            "groundtruth_waypoints": gt_waypoints,
            "meta": {
                "base_model": args.base_model,
                "disable_lora": args.disable_lora,
                "adapter_path": None if args.disable_lora else args.adapter_path,
                "max_new_tokens": args.max_new_tokens,
                "use_masked_prompt": args.use_masked_prompt,
            }
        }

        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Saved inference record to: {args.save_json}")


if __name__ == "__main__":
    main()
