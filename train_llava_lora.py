import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, TaskType


IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a helpful and socially-aware planning module for understanding "
    "urban driving scenes, social cues, and outputting future waypoints "
    "for the ego vehicle."
)

USER_PROMPT_BASE = (
    "What should the ego vehicle do next? Reasoning the plan. "
    "And plan a pixel-wise future trajectory in the format [(x1, y1), (x2, y2), ...]. "
    "Social cue: "
)


def resolve_image_path(image_root: str, image_path_in_json: str) -> str:
    if image_path_in_json is None:
        raise ValueError("image path is None")

    p = str(image_path_in_json)
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(image_root, p)


def format_waypoints(gt_traj: List[List[float]]) -> str:
    pairs = ", ".join([f"({int(x)}, {int(y)})" for x, y in gt_traj])
    return f"[{pairs}]"


def build_answer(gt_traj: List[List[float]], reasoning: str) -> str:
    wps = format_waypoints(gt_traj)
    reasoning = (reasoning or "").strip()
    if len(reasoning) == 0:
        reasoning = "No reasoning provided."
    return f"Waypoints: {wps}\nReasoning: {reasoning}"


def build_llava_prompt(system_prompt: str, user_prompt: str) -> str:
    # Stable manual prompt format to avoid chat_template mismatch
    return (
        f"{system_prompt}\n"
        f"USER: <image>\n{user_prompt}\n"
        f"ASSISTANT:"
    )


def _get_key(x: Dict[str, Any]):
    return x.get("clip_id") or x.get("id") or x.get("image") or x.get("img") or x.get("image_path")


def _get_text0(x: Dict[str, Any]) -> Dict[str, Any]:
    return (x.get("texts") or [{}])[0]


def _get_image_path(x: Dict[str, Any]) -> Optional[str]:
    t0 = _get_text0(x)
    return x.get("image") or x.get("img") or x.get("image_path") or t0.get("image") or t0.get("img") or t0.get("image_path")


class TitanPairedDataset(Dataset):
    """
    Train dataset: expects cue/no_cue JSON files with matching clip_id.
    Each item returns:
      - user_full: cue prompt
      - user_mask: masked (no-cue) prompt
      - same image
      - same target answer
    """

    def __init__(
        self,
        cue_json_path: str,
        nocue_json_path: str,
        image_root: str,
        max_train_samples: Optional[int] = None,
    ):
        self.image_root = image_root
        self.samples: List[Dict[str, Any]] = []

        with open(cue_json_path, "r", encoding="utf-8") as f:
            cue_data = json.load(f)
        with open(nocue_json_path, "r", encoding="utf-8") as f:
            nocue_data = json.load(f)

        cue_by_id = {(_get_key(x)): x for x in cue_data if _get_key(x) is not None}
        nocue_by_id = {(_get_key(x)): x for x in nocue_data if _get_key(x) is not None}
        common_ids = sorted(set(cue_by_id.keys()) & set(nocue_by_id.keys()))

        for cid in common_ids:
            c = cue_by_id[cid]
            n = nocue_by_id[cid]

            ct = _get_text0(c)
            nt = _get_text0(n)

            gt_traj = ct.get("gt_traj", None)
            reasoning = ct.get("assistant", "")

            social_cue_full = ct.get("social_cue", "")
            social_cue_mask = nt.get("social_cue", "")

            if gt_traj is None or not isinstance(gt_traj, list) or len(gt_traj) == 0:
                continue

            image_path = _get_image_path(c)
            if image_path is None:
                continue

            user_full = USER_PROMPT_BASE + (social_cue_full or "")
            user_mask = USER_PROMPT_BASE + (social_cue_mask or "")
            answer = build_answer(gt_traj, reasoning)

            self.samples.append(
                {
                    "clip_id": cid,
                    "image": image_path,
                    "user_full": user_full,
                    "user_mask": user_mask,
                    "answer": answer,
                }
            )

        if max_train_samples is not None and max_train_samples > 0:
            self.samples = self.samples[:max_train_samples]

        print(f"[TitanPairedDataset] paired samples kept = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = resolve_image_path(self.image_root, sample["image"])
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "user_full": sample["user_full"],
            "user_mask": sample["user_mask"],
            "answer": sample["answer"],
            "clip_id": sample["clip_id"],
        }


class TitanCueOnlyDataset(Dataset):
    """
    Eval dataset: only cue JSON.
    We synthesize masked prompt by clearing social cue (or you can choose another masking rule).
    """

    def __init__(
        self,
        cue_json_path: str,
        image_root: str,
        max_samples: Optional[int] = None,
        mask_strategy: str = "empty",  # currently only "empty"
    ):
        self.image_root = image_root
        self.samples: List[Dict[str, Any]] = []

        with open(cue_json_path, "r", encoding="utf-8") as f:
            cue_data = json.load(f)

        for c in cue_data:
            cid = _get_key(c)
            ct = _get_text0(c)

            gt_traj = ct.get("gt_traj", None)
            reasoning = ct.get("assistant", "")
            social_cue_full = ct.get("social_cue", "")

            if gt_traj is None or not isinstance(gt_traj, list) or len(gt_traj) == 0:
                continue

            image_path = _get_image_path(c)
            if image_path is None:
                continue

            user_full = USER_PROMPT_BASE + (social_cue_full or "")
            if mask_strategy == "empty":
                user_mask = USER_PROMPT_BASE  # social cue cleared
            else:
                user_mask = USER_PROMPT_BASE

            answer = build_answer(gt_traj, reasoning)

            self.samples.append(
                {
                    "clip_id": cid,
                    "image": image_path,
                    "user_full": user_full,
                    "user_mask": user_mask,
                    "answer": answer,
                }
            )

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        print(f"[TitanCueOnlyDataset] samples kept = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = resolve_image_path(self.image_root, sample["image"])
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "user_full": sample["user_full"],
            "user_mask": sample["user_mask"],
            "answer": sample["answer"],
            "clip_id": sample["clip_id"],
        }


@dataclass
class PairedDataCollator:
    processor: LlavaProcessor
    max_length: int
    system_prompt: str

    def _tokenize_one(self, image: Image.Image, prompt_text: str, answer_text: str) -> Dict[str, torch.Tensor]:
        # Prompt-only encoding (to mask labels on prompt part)
        enc_prompt = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        # Prompt + answer
        full_text = prompt_text + " " + answer_text
        enc_full = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        pixel_values = enc_full["pixel_values"][0]

        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[0])
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_items = []
        mask_items = []

        for ex in features:
            image = ex["image"]
            answer = ex["answer"]

            prompt_full = build_llava_prompt(self.system_prompt, ex["user_full"])
            prompt_mask = build_llava_prompt(self.system_prompt, ex["user_mask"])

            full_items.append(self._tokenize_one(image, prompt_full, answer))
            mask_items.append(self._tokenize_one(image, prompt_mask, answer))

        def pad_1d(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
            max_len = max(x.shape[0] for x in seqs)
            out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
            for i, x in enumerate(seqs):
                out[i, : x.shape[0]] = x
            return out

        batch_full = {
            "input_ids": pad_1d([x["input_ids"] for x in full_items], self.processor.tokenizer.pad_token_id),
            "attention_mask": pad_1d([x["attention_mask"] for x in full_items], 0),
            "labels": pad_1d([x["labels"] for x in full_items], IGNORE_INDEX),
            "pixel_values": torch.stack([x["pixel_values"] for x in full_items], dim=0),
        }

        batch_mask = {
            "input_ids": pad_1d([x["input_ids"] for x in mask_items], self.processor.tokenizer.pad_token_id),
            "attention_mask": pad_1d([x["attention_mask"] for x in mask_items], 0),
            "labels": pad_1d([x["labels"] for x in mask_items], IGNORE_INDEX),
            "pixel_values": torch.stack([x["pixel_values"] for x in mask_items], dim=0),
        }

        return {"full": batch_full, "masked": batch_mask}


def masked_kl_div_on_labeled_tokens(
    logits_full: torch.Tensor,
    logits_masked: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    valid = labels.ne(IGNORE_INDEX)
    if valid.sum() == 0:
        return logits_full.new_tensor(0.0)

    T = temperature
    log_p = F.log_softmax(logits_full / T, dim=-1)
    log_q = F.log_softmax(logits_masked / T, dim=-1)
    p = F.softmax(logits_full / T, dim=-1)

    kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [B, T]
    kl = kl_per_token[valid].mean() * (T * T)
    return kl


class PairedKLTrainer(Trainer):
    def __init__(self, *args, lambda_kl: float = 0.3, kl_temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_kl = lambda_kl
        self.kl_temperature = kl_temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        full = inputs["full"]
        masked = inputs["masked"]

        out_full = model(
            input_ids=full["input_ids"],
            attention_mask=full["attention_mask"],
            pixel_values=full["pixel_values"],
            labels=full["labels"],
            use_cache=False,
            return_dict=True,
        )
        ce_loss = out_full.loss

        out_mask = model(
            input_ids=masked["input_ids"],
            attention_mask=masked["attention_mask"],
            pixel_values=masked["pixel_values"],
            labels=None,
            use_cache=False,
            return_dict=True,
        )

        logits_full = out_full.logits
        logits_mask = out_mask.logits
        labels_full = full["labels"]

        min_t = min(logits_full.shape[1], logits_mask.shape[1], labels_full.shape[1])
        logits_full = logits_full[:, :min_t, :]
        logits_mask = logits_mask[:, :min_t, :]
        labels_full = labels_full[:, :min_t]

        kl_loss = masked_kl_div_on_labeled_tokens(
            logits_full=logits_full,
            logits_masked=logits_mask,
            labels=labels_full,
            temperature=self.kl_temperature,
        )

        loss = ce_loss + self.lambda_kl * kl_loss

        # logging（train/eval 都會出現）
        self.log(
            {
                "loss_ce": float(ce_loss.detach().cpu()),
                "loss_kl": float(kl_loss.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
            }
        )

        if return_outputs:
            return loss, {"out_full": out_full, "out_mask": out_mask}
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        # 讓 evaluation 也用同一套 compute_loss (CE + KL)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # 我們不需要 logits / labels（省記憶體也避免 shape 對不上）
        return (loss.detach(), None, None)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--train_cue", type=str, required=True)
    p.add_argument("--train_nocue", type=str, required=True)

    # ✅ eval only needs cue json
    p.add_argument("--eval_cue", type=str, default=None)

    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)

    p.add_argument("--lambda_kl", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- Processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- Model dtype
    use_bf16 = bool(args.bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ---- Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Train dataset (paired)
    train_dataset = TitanPairedDataset(
        cue_json_path=args.train_cue,
        nocue_json_path=args.train_nocue,
        image_root=args.image_root,
        max_train_samples=args.max_train_samples,
    )
    if len(train_dataset) == 0:
        raise ValueError("No valid paired samples found in train set.")

    # ---- Eval dataset (cue only)
    eval_dataset = None
    if args.eval_cue is not None:
        eval_dataset = TitanCueOnlyDataset(
            cue_json_path=args.eval_cue,
            image_root=args.image_root,
            max_samples=args.max_eval_samples,
            mask_strategy="empty",
        )
        if len(eval_dataset) == 0:
            print("[WARN] eval_cue provided but no valid samples kept. Disable evaluation.")
            eval_dataset = None

    # ---- Data collator
    data_collator = PairedDataCollator(
        processor=processor,
        max_length=args.max_length,
        system_prompt=SYSTEM_PROMPT,
    )

    # ---- Training arguments
    do_eval = eval_dataset is not None
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        bf16=bool(args.bf16),
        fp16=(not bool(args.bf16)) if (args.fp16 or not args.bf16) else False,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,

        save_strategy="steps",
        logging_strategy="steps",
        evaluation_strategy=("steps" if do_eval else "no"),

        # 如果你想用 val loss 挑 best ckpt，打開這三行：
        load_best_model_at_end=(True if do_eval else False),
        metric_for_best_model=("eval_loss" if do_eval else None),
        greater_is_better=False if do_eval else None,

        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        prediction_loss_only=True,  
    )

    trainer = PairedKLTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        lambda_kl=args.lambda_kl,
        kl_temperature=args.temperature,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)

    print(f"Training finished. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()