#!/usr/bin/env python3
"""
batch_infer_eval.py

Batch inference + evaluation with ONE-TIME model load.

Behavior:
- If --use_lora is set (default), runs inference using functions from infer_llava_lora.py
- If --use_lora is NOT set, runs ICL inference using functions from infer_icl.py
- After inference, runs evaluation in-process using evaluation_sweep.py

Example (LoRA):
python batch_infer_eval.py \
  --use_lora \
  --base_model llava-hf/llava-1.5-7b-hf \
  --adapter_path ./ckpt_xxx \
  --image_root dataset/images_anonymized \
  --cue_json ./titan_train.json \
  --nocue_json ./titan_train_no_cue.json \
  --num_samples 128 \
  --seed 42 \
  --out_pred_json ./pred_batch.json \
  --out_eval_json ./eval_batch.json \
  --reasoning_only_text_metrics \
  --l2_mode max \
  --resolution_sweep "1920x1080,1280x720,960x540,640x360,512x288"

Example (ICL, no LoRA):
python batch_infer_eval.py \
  --no_lora \
  --base_model llava-hf/llava-1.5-7b-hf \
  --image_root dataset/images_anonymized \
  --cue_json ./titan_train.json \
  --nocue_json ./titan_train_no_cue.json \
  --icl_k 3 --icl_seed 42 --icl_from cue \
  --num_samples 128 --seed 42 \
  --out_pred_json ./pred_batch_icl.json \
  --out_eval_json ./eval_batch_icl.json
"""
import os
import json
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Local modules (the files you uploaded)
import infer_llava_lora
import infer_icl
import evaluation_sweep

from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_pad(processor: LlavaProcessor, model: LlavaForConditionalGeneration):
    # keep consistent with your infer scripts
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id


def _load_processor(base_model: str) -> LlavaProcessor:
    # match infer_llava_lora: manual construction for compatibility
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(base_model)
    return LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)


def _load_model_once(
    base_model: str,
    device: str,
    dtype: torch.dtype,
    use_lora: bool,
    adapter_path: Optional[str],
) -> Tuple[torch.nn.Module, LlavaProcessor]:
    processor = _load_processor(base_model)

    base = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    _ensure_pad(processor, base)

    if use_lora:
        if not adapter_path:
            raise ValueError("--adapter_path is required when --use_lora is set")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model.eval()
    return model, processor


def _make_gt_answer(gt_waypoints: Any, gt_reasoning: str) -> str:
    wp_txt = infer_llava_lora.format_waypoints_for_demo(gt_waypoints)
    return f"Waypoints: {wp_txt}\nReasoning: {gt_reasoning or ''}".strip()


def _save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _select_query_indices(n_total: int, num_samples: int, seed: int) -> List[int]:
    if num_samples <= 0 or num_samples >= n_total:
        return list(range(n_total))
    rnd = random.Random(seed)
    return rnd.sample(list(range(n_total)), num_samples)


def _infer_batch_lora(
    model,
    processor,
    samples: List[Dict[str, Any]],
    image_root: str,
    max_new_tokens: int,
    device: str,
    use_masked_prompt: bool,
    run_tag: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for s in samples:
        image_path = infer_llava_lora.resolve_image_path(image_root, s["image_path"])
        social_text = s["nocue_text"] if use_masked_prompt else s["cue_text"]

        prompt = infer_llava_lora.build_llava_prompt(
            infer_llava_lora.SYSTEM_PROMPT,
            infer_llava_lora.build_basic_user_prompt(social_text),
        )

        image = Image.open(image_path).convert("RGB")
        out_text = infer_llava_lora.generate_one(
            model=model,
            processor=processor,
            image=image,
            prompt_text=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        pred_wps = infer_llava_lora.parse_waypoints_tagged(out_text)
        gt_wps = s.get("gt_traj")
        gt_reasoning = s.get("reasoning_gt", "")

        rec = {
            "sample_id": str(s.get("key")),
            "image": image_path,
            "mode": "masked" if use_masked_prompt else "cue",
            "predicted_answer": out_text,
            "groundtruth_answer": _make_gt_answer(gt_wps, gt_reasoning),
            "predicted_waypoints": pred_wps,
            "groundtruth_waypoints": gt_wps,
            "meta": {
                "runner": "batch_infer_eval.py",
                "run_tag": run_tag,
                "use_lora": True,
                "max_new_tokens": max_new_tokens,
                "use_masked_prompt": use_masked_prompt,
            },
        }
        records.append(rec)
    return records


def _build_prompt_and_images_for_icl(
    query: Dict[str, Any],
    icl_examples: List[Dict[str, Any]],
    image_root: str,
    icl_from: str,
    text_only_icl: bool,
    use_masked_prompt: bool,
) -> Tuple[str, List[Image.Image]]:
    query_image_path = infer_icl.resolve_image_path(image_root, query["image_path"])
    query_social_text = query["nocue_text"] if use_masked_prompt else query["cue_text"]

    images: List[Image.Image] = []

    if text_only_icl:
        lines: List[str] = []
        if icl_examples:
            lines.append("Here are example input-output pairs showing the expected format and behavior.\n")
            for i, ex in enumerate(icl_examples, 1):
                social_text = ex.get("cue_text", "") if icl_from == "cue" else ex.get("nocue_text", "")
                user_part = infer_icl.build_basic_user_prompt(social_text)
                assistant_part = infer_icl.build_output_from_example(ex)
                lines.append(f"[Example {i}]")
                lines.append(f"User: {user_part}")
                lines.append(f"Assistant: {assistant_part}\n")
        icl_block = "\n".join(lines)
        query_user_prompt = infer_icl.build_basic_user_prompt(query_social_text)

        full_user_prompt = (icl_block + "[Now solve the following case]\n" + query_user_prompt) if icl_block else query_user_prompt
        prompt_text = f"{infer_icl.SYSTEM_PROMPT}\nUSER: <image>\n{full_user_prompt}\nASSISTANT:"

        # only 1 image
        images.append(Image.open(query_image_path).convert("RGB"))
        return prompt_text, images

    # Multi-image True ICL format
    prompt_text = infer_icl.SYSTEM_PROMPT + "\n"
    for ex in icl_examples:
        ex_img_path = infer_icl.resolve_image_path(image_root, ex["image_path"])
        images.append(Image.open(ex_img_path).convert("RGB"))

        social_text = ex.get("cue_text", "") if icl_from == "cue" else ex.get("nocue_text", "")
        user_part = infer_icl.build_basic_user_prompt(social_text)
        assistant_part = infer_icl.build_output_from_example(ex)

        prompt_text += f"USER: <image>\n{user_part}\nASSISTANT: {assistant_part}</s>"

    images.append(Image.open(query_image_path).convert("RGB"))
    query_user_prompt = infer_icl.build_basic_user_prompt(query_social_text)
    prompt_text += f"USER: <image>\n{query_user_prompt}\nASSISTANT:"
    return prompt_text, images


def _infer_batch_icl(
    model,
    processor,
    all_pairs: List[Dict[str, Any]],
    query_samples: List[Dict[str, Any]],
    image_root: str,
    max_new_tokens: int,
    device: str,
    use_masked_prompt: bool,
    icl_k: int,
    icl_seed: int,
    icl_from: str,
    text_only_icl: bool,
    run_tag: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for s in query_samples:
        icl_examples = infer_icl.sample_icl_examples(
            all_pairs=all_pairs,
            query_key=s.get("key"),
            k=icl_k,
            seed=icl_seed,
        )

        prompt_text, images = _build_prompt_and_images_for_icl(
            query=s,
            icl_examples=icl_examples,
            image_root=image_root,
            icl_from=icl_from,
            text_only_icl=text_only_icl,
            use_masked_prompt=use_masked_prompt,
        )

        out_text = infer_icl.generate_one(
            model=model,
            processor=processor,
            images=images,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        pred_wps = infer_icl.parse_waypoints_tagged(out_text)
        gt_wps = s.get("gt_traj")
        gt_reasoning = s.get("reasoning_gt", "")

        rec = {
            "sample_id": str(s.get("key")),
            "image": infer_icl.resolve_image_path(image_root, s["image_path"]),
            "mode": "masked" if use_masked_prompt else "cue",
            "predicted_answer": out_text,
            "groundtruth_answer": _make_gt_answer(gt_wps, gt_reasoning),
            "predicted_waypoints": pred_wps,
            "groundtruth_waypoints": gt_wps,
            "meta": {
                "runner": "batch_infer_eval.py",
                "run_tag": run_tag,
                "use_lora": False,
                "icl_k": icl_k,
                "icl_seed": icl_seed,
                "icl_from": icl_from,
                "text_only_icl": text_only_icl,
                "max_new_tokens": max_new_tokens,
                "use_masked_prompt": use_masked_prompt,
            },
        }
        records.append(rec)

    return records


def _run_eval(
    records: List[Dict[str, Any]],
    out_eval_json: str,
    l2_mode: str,
    reasoning_only_text_metrics: bool,
    align_mode: str,
    resolution_sweep: Optional[str],
    image_root: str,
):
    sweep_list = evaluation_sweep.parse_wh_list(resolution_sweep) if resolution_sweep else None
    results = []
    for rec in records:
        try:
            r = evaluation_sweep.evaluate_one_sample(
                rec,
                l2_mode=l2_mode,
                use_reasoning_only_for_text_metrics=reasoning_only_text_metrics,
                use_gpt4o_grader=False,
                gpt4o_model="gpt-4o",
                align_mode=align_mode,
                resolution_sweep=sweep_list,
                image_root=image_root,
            )
            results.append(r)
        except Exception as e:
            results.append({"sample_id": rec.get("sample_id"), "error": str(e)})

    summary = evaluation_sweep.aggregate_results([r for r in results if "error" not in r])
    payload = {"summary": summary, "results": results}
    _save_json(out_eval_json, payload)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")

    # choose runner
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--use_lora", action="store_true", help="Use LoRA weights (infer_llava_lora style). Default if neither is set.")
    group.add_argument("--no_lora", action="store_true", help="Do NOT use LoRA weights; use ICL (infer_icl style).")
    ap.add_argument("--adapter_path", type=str, default=None)

    # data
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--cue_json", type=str, required=True)
    ap.add_argument("--nocue_json", type=str, required=True)

    # batching
    ap.add_argument("--num_samples", type=int, default=0, help="0 means all paired samples")
    ap.add_argument("--seed", type=int, default=42, help="Seed for selecting query samples")
    ap.add_argument("--use_masked_prompt", action="store_true", help="Query uses no-cue social text (mode=masked)")

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # ICL settings (only used when --no_lora)
    ap.add_argument("--icl_k", type=int, default=2)
    ap.add_argument("--icl_seed", type=int, default=42)
    ap.add_argument("--icl_from", type=str, default="cue", choices=["cue", "nocue"])
    ap.add_argument("--text_only_icl", action="store_true")

    # output
    ap.add_argument("--out_pred_json", type=str, required=True)
    ap.add_argument("--out_eval_json", type=str, required=True)
    ap.add_argument("--run_tag", type=str, default="")

    # evaluation options
    ap.add_argument("--l2_mode", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--align_mode", type=str, default="min", choices=["min", "gt"])
    ap.add_argument("--reasoning_only_text_metrics", action="store_true")
    ap.add_argument("--resolution_sweep", type=str, default="640x360",
                    help='Comma-separated resolutions, e.g. "1920x1080,1280x720,640x360"')

    args = ap.parse_args()

    # default behavior: use_lora unless explicitly --no_lora
    use_lora = args.use_lora or (not args.no_lora)

    _set_seed(args.seed)

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

    print(f"[INFO] device={device}, dtype={dtype}, use_lora={use_lora}")

    # Load paired samples once
    all_pairs = infer_llava_lora.load_all_paired_samples(args.cue_json, args.nocue_json)
    idxs = _select_query_indices(len(all_pairs), args.num_samples, args.seed)
    query_samples = [all_pairs[i] for i in idxs]
    print(f"[INFO] paired samples={len(all_pairs)}, query_samples={len(query_samples)}")

    # Load model/processor ONCE
    model, processor = _load_model_once(
        base_model=args.base_model,
        device=device,
        dtype=dtype,
        use_lora=use_lora,
        adapter_path=args.adapter_path,
    )

    # Inference
    if use_lora:
        records = _infer_batch_lora(
            model=model,
            processor=processor,
            samples=query_samples,
            image_root=args.image_root,
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_masked_prompt=args.use_masked_prompt,
            run_tag=args.run_tag,
        )
    else:
        records = _infer_batch_icl(
            model=model,
            processor=processor,
            all_pairs=all_pairs,
            query_samples=query_samples,
            image_root=args.image_root,
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_masked_prompt=args.use_masked_prompt,
            icl_k=args.icl_k,
            icl_seed=args.icl_seed,
            icl_from=args.icl_from,
            text_only_icl=args.text_only_icl,
            run_tag=args.run_tag,
        )

    _save_json(args.out_pred_json, records)
    print(f"[INFO] Saved predictions to: {args.out_pred_json}")

    # Evaluation
    summary = _run_eval(
        records=records,
        out_eval_json=args.out_eval_json,
        l2_mode=args.l2_mode,
        reasoning_only_text_metrics=args.reasoning_only_text_metrics,
        align_mode=args.align_mode,
        resolution_sweep=args.resolution_sweep,
        image_root="",  # records already store absolute image paths
    )
    print("==== Evaluation Summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved eval to: {args.out_eval_json}")


if __name__ == "__main__":
    main()
