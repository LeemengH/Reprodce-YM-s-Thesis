# Reprodce-YM-s-Thesis
## Trajectory Planning in Dense Urban Environments: Utilizing Vision-Language Models to Learn Socially-Aware Behaviors for High-Uncertainty Scenarios
This repository documents the reproduction of my senior’s work on trajectory-aware fine-tuning of **LLaVA-1.5-7B** for dense urban autonomous driving scenarios (TITAN dataset).

The goal of this reproduction is:

- ✅ Reconstruct spatial supervision via COLMAP
- ✅ Fine-tune LLaVA with KL-regularized LoRA
- ✅ Reproduce reasoning + waypoint prediction performance
- ✅ Establish a stable baseline for future vision token pruning research

---

# 1. Environment Setup

## 1.1 Conda Environment

```bash
conda create -n llava_kl_lora python=3.10 -y
conda activate llava_kl_lora
pip install -U pip setuptools wheel
````

---

## 1.2 PyTorch Installation

### ⚠ For RTX 5090 (CUDA 12.8 nightly)

```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Otherwise (A6000 / 3090 etc.)

Install the stable CUDA version matching your driver.

---

## 1.3 Required Python Packages

```bash
pip install \
  "transformers==4.41.2" \
  "accelerate==0.31.0" \
  "peft==0.11.1" \
  "tokenizers==0.19.1" \
  "huggingface_hub>=0.23,<1.0" \
  "protobuf" \
  "sentencepiece" \
  "datasets==2.20.0" \
  "pillow" \
  "opencv-python" \
  "tqdm" \
  "numpy" \
  "scipy"
```

---

# 2. Project Folder Structure

```
ym_reproduce/
│
├── train_llava_lora.py
├── infer_llava_lora.py
├── evaluation.py
│
├── dataset/
│   └── images_anonymized/
│       └── clip_x/
│           └── images/
│
├── fill_json/
│   ├── titan_train_filled.json
│   └── titan_train_no_cue.json
│
├── ckpt_kl_lambda0p01_bf16/
│
└── results/
```

---

# 3. Fine-tuning Details

## 3.1 Model Backbone

* Base Model: `llava-hf/llava-1.5-7b-hf`
* Fine-tuning method: LoRA
* Loss:

  * Cross-entropy (reasoning + waypoint tokens)
  * KL regularization to control hallucination
* Output format:

  * Reasoning (CoT)
  * 20 (x, y) waypoint pairs

---

## 3.2 Training Command

```bash
python train_llava_lora.py \
  --model_name llava-hf/llava-1.5-7b-hf \
  --train_cue ./fill_json/titan_train_filled.json \
  --train_nocue ./fill_json/titan_train_no_cue.json \
  --image_root dataset/images_anonymized \
  --output_dir ./ckpt_kl_lambda0p01_bf16 \
  --max_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --lambda_kl 0.01 \
  --temperature 1.0 \
  --bf16 \
  --gradient_checkpointing
```

---

## 3.3 Key Hyperparameters

| Parameter         | Value   | Description             |
| ----------------- | ------- | ----------------------- |
| batch size        | 2       | per GPU                 |
| grad accumulation | 16      | effective batch = 32    |
| max length        | 4096    | supports long CoT       |
| learning rate     | 2e-4    | LoRA training           |
| lambda_kl         | 0.01    | controls hallucination  |
| bf16              | enabled | A6000 / 5090 compatible |

---

# 4. Evaluation

Evaluation includes:

* Text metrics (reasoning quality)
* Waypoint L2 error
* Waypoint format success rate

---

## 4.1 Example Result

```json
{
  "bleu4": 0.3312288595914515,
  "rougeL_f1": 0.39126456154892464,
  "meteor": 0.5368330601352392,
  "l2_distance_error": ...
}
```

---

## 4.2 Qualitative Visualization (2×2 Grid)

You can replace the image paths with your own result images.

<table>
<tr>
<td align="center"><img src="results/clip1_pred.png" width="400"><br>Prediction 1</td>
<td align="center"><img src="results/clip1_gt.png" width="400"><br>Ground Truth 1</td>
</tr>
<tr>
<td align="center"><img src="results/clip2_pred.png" width="400"><br>Prediction 2</td>
<td align="center"><img src="results/clip2_gt.png" width="400"><br>Ground Truth 2</td>
</tr>
</table>

---

## 4.3 Comparison with Senior’s Reported Results

| Metric   | Senior | Reproduced | Notes (w/o Fine-tune)|
| -------- | ------ | ---------- | ----- |
| BLEU-4   | 0.036  | 0.083      |       |
| ROUGE-L  | 0.19   | 0.146      |       |
| METEOR   | 0.37   | 0.197      |       |
| L2 Error | 55     | 175        |       |

| Metric   | Senior | Reproduced | Notes (w/ Fine-tune)|
| -------- | ------ | ---------- | ----- |
| BLEU-4   | 0.18   | 0.3312     |       |
| ROUGE-L  | 0.38   | 0.3912     |       |
| METEOR   | 0.52   | 0.5368     |       |
| L2 Error | 31     | 51         |       |

---

# Appendix

# A. Reconstructing Missing Trajectories with COLMAP

Some clips do not contain ground truth camera trajectories.
We reconstruct spatial supervision using:

* Structure-from-Motion (SfM)
* Camera intrinsics & extrinsics
* Keyframe projection of future lookahead points

---

## A.1 COLMAP Pipeline

1. Extract images per clip
2. Feature extraction
3. Feature matching
4. Sparse reconstruction
5. Export:

   * `cameras.txt`
   * `images.txt`

---

## A.2 Projection Method

For each keyframe:

1. Convert quaternion → rotation matrix
2. Compute camera center:

```
C = -R^T t
```

3. Generate lookahead world points
4. Project 3D → 2D onto keyframe
5. Normalize to image resolution

---

## A.3 Design Decision

* Follow senior’s projection-based approach
* No IMU correction used
* All spatial supervision derived from SfM

This ensures:

* Interpretability
* Resolution-aware L2 comparison
* Consistent evaluation pipeline

---

# Reproduction Philosophy

This reproduction strictly follows the original methodology to:

* Match output format (reasoning + 20 waypoints)
* Reproduce trajectory accuracy
* Establish a stable baseline for future pruning experiments

This serves as the foundation for subsequent lightweighting and vision token pruning research.

```
```
