# Smart Cropping Pipeline - 3 Stage Workflow

A high-performance, multiprocessing pipeline for intelligently cropping large datasets using DINOv2/v3 models.

## Overview

This pipeline splits the smart cropping workflow into 3 separate stages for maximum efficiency:

1. **Stage 1 (CPU)**: Preprocess images, generate crop candidates, and save tensors
2. **Stage 2 (GPU)**: Run inference to identify best crops based on semantic similarity
3. **Stage 3 (CPU)**: Save the final cropped images

### Why 3 Stages?

- **Separation of concerns**: CPU-intensive preprocessing is separated from GPU inference
- **Parallelization**: Each stage can leverage multiprocessing/multi-GPU independently
- **Flexibility**: Re-run any stage without repeating previous work
- **Resource optimization**: Use all CPU cores for preprocessing/saving, GPUs only for inference
- **Debugging**: Easier to debug and monitor each stage separately

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           STAGE 1 (CPU)                              │
│                    Multiprocessing Preprocessing                     │
├─────────────────────────────────────────────────────────────────────┤
│  Input: HuggingFace Dataset                                         │
│  Process:                                                            │
│    1. Load image from dataset                                       │
│    2. Resize to target size (e.g., 1024px)                          │
│    3. Generate N crop candidates (e.g., 15 crops)                   │
│    4. Apply model preprocessing (resize to 224x224, normalize)      │
│    5. Save tensors: [num_images, C, H, W] + metadata                │
│  Output: .pt files (preprocessed_tensors/*.pt)                      │
│    - tensors: Normalized model inputs [N+1, 3, 224, 224]            │
│    - original_crops: Original-sized crops [N+1, 3, H, W]            │
│    - crop_boxes: List of crop coordinates                           │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                           STAGE 2 (GPU)                              │
│                         Batched Inference                            │
├─────────────────────────────────────────────────────────────────────┤
│  Input: Preprocessed tensors (.pt files)                            │
│  Process:                                                            │
│    1. Load preprocessed tensors from disk                           │
│    2. Batch multiple images together                                │
│    3. Run DINOv2/v3 inference on GPU                                │
│    4. Compute cosine similarity between original and crops          │
│    5. Identify best crop for each image                             │
│  Output: crop_metadata.json                                         │
│    - image_id, best_crop_idx, crop_box, similarity, etc.            │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                           STAGE 3 (CPU)                              │
│                     Multiprocessing Save Crops                       │
├─────────────────────────────────────────────────────────────────────┤
│  Input: Metadata (JSON) + Preprocessed tensors (.pt)                │
│  Process:                                                            │
│    1. Read metadata to get best crop index                          │
│    2. Load original-sized crop tensor from Stage 1                  │
│    3. Convert tensor to PIL Image                                   │
│    4. Save as PNG/JPG                                                │
│  Output: Cropped images (cropped_images/*.png)                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install torch torchvision transformers datasets accelerate pillow tqdm
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
python run_pipeline.py \
    --dataset ujin-song/pexels-image-60k \
    --split train \
    --num_crops 15 \
    --resize_to 1024 \
    --model_name facebook/dinov2-base \
    --batch_size 8 \
    --max_samples 100
```

### Option 2: Run Stages Individually

**Stage 1: Preprocess**
```bash
python stage1_preprocess.py \
    --dataset ujin-song/pexels-image-60k \
    --split train \
    --output_dir ./preprocessed_tensors \
    --num_crops 15 \
    --resize_to 1024 \
    --model_input_size 224 \
    --num_workers 16 \
    --max_samples 100
```

**Stage 2: Inference**
```bash
python stage2_inference.py \
    --tensor_dir ./preprocessed_tensors \
    --output_file ./crop_metadata.json \
    --model_name facebook/dinov2-base \
    --batch_size 8 \
    --device cuda \
    --num_workers 4
```

**Stage 3: Save Crops (Optimized)**
```bash
python stage3_save_crops_optimized.py \
    --metadata_file ./crop_metadata.json \
    --tensor_dir ./preprocessed_tensors \
    --output_dir ./cropped_images \
    --num_workers 16
```

## Stage Details

### Stage 1: CPU Multiprocessing Preprocessing

**Purpose**: Prepare images and generate preprocessed tensors

**Key Parameters**:
- `--num_crops`: Number of crop candidates per image (default: 15)
- `--resize_to`: Resize shorter side to this value (default: 1024)
- `--model_input_size`: Model input size, e.g., 224 for DINOv2 (default: 224)
- `--num_workers`: CPU workers for multiprocessing (default: cpu_count)

**Output Format** (.pt files):
```python
{
    "image_id": "image_001",
    "tensors": torch.Tensor,           # [N+1, 3, 224, 224] normalized for model
    "original_crops": [torch.Tensor],  # List of [3, H, W] original-sized crops
    "crop_boxes": [(left, top, right, bottom), ...],
    "original_size": (width, height)
}
```

**Performance**: Scales linearly with CPU cores

### Stage 2: GPU Inference

**Purpose**: Run model inference to identify best crops

**Key Parameters**:
- `--model_name`: DINOv2/v3 model (default: facebook/dinov2-base)
- `--batch_size`: Images per batch (default: 8)
- `--device`: cuda or cpu (default: cuda)
- `--num_workers`: DataLoader workers (default: 4)

**Output Format** (crop_metadata.json):
```json
[
  {
    "image_id": "image_001",
    "crop_box": [100, 0, 1124, 1024],
    "similarity": 0.9234,
    "best_crop_idx": 3,
    "original_size": [1920, 1024]
  }
]
```

**Performance**:
- Effective batch size: `batch_size * (num_crops + 1)`
- Example: 8 images × 16 crops = 128 tensors per batch
- Supports multi-GPU via PyTorch DataParallel

### Stage 3: Save Cropped Images

**Two Versions Available**:

#### Optimized Version (Recommended)
`stage3_save_crops_optimized.py`
- Reads original-sized crops from Stage 1 tensors
- No need to reload dataset
- Faster and more efficient

```bash
python stage3_save_crops_optimized.py \
    --metadata_file ./crop_metadata.json \
    --tensor_dir ./preprocessed_tensors \
    --output_dir ./cropped_images
```

#### Standard Version
`stage3_save_crops.py`
- Reloads images from HuggingFace dataset
- Applies crop boxes to original images
- Use if you modified Stage 1 to not save original_crops

```bash
python stage3_save_crops.py \
    --metadata_file ./crop_metadata.json \
    --dataset ujin-song/pexels-image-60k \
    --split train \
    --output_dir ./cropped_images
```

**Performance**: Scales linearly with CPU cores

## Use Cases

### 1. Processing Subset of Dataset

```bash
python run_pipeline.py \
    --start_idx 1000 \
    --max_samples 500 \
    --dataset my-dataset
```

### 2. Re-run Only Inference (After Changing Model)

```bash
# Preprocess once
python stage1_preprocess.py --dataset my-dataset

# Try different models
python stage2_inference.py --model_name facebook/dinov2-base
python stage2_inference.py --model_name facebook/dinov2-large --output_file metadata_large.json

# Save crops from best model
python stage3_save_crops_optimized.py --metadata_file metadata_large.json
```

### 3. Skip Stages (Resume Pipeline)

```bash
# Already ran Stage 1 and 2, only run Stage 3
python run_pipeline.py \
    --skip_stage1 \
    --skip_stage2 \
    --use_optimized_stage3
```

### 4. Multi-GPU Inference

For Stage 2, PyTorch will automatically use all available GPUs if you set:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python stage2_inference.py --batch_size 32
```

## Performance Tips

1. **Stage 1 Optimization**:
   - Use `--num_workers` = CPU cores
   - Adjust `--resize_to` based on GPU memory
   - Fewer crops = faster but less accurate

2. **Stage 2 Optimization**:
   - Increase `--batch_size` until GPU memory is full
   - Use smaller models (dinov2-small/vits) for speed
   - Effective batch = `batch_size × (num_crops + 1)`

3. **Stage 3 Optimization**:
   - Always use optimized version
   - Use `--num_workers` = CPU cores
   - Save as JPEG instead of PNG for smaller files (modify code)

## Disk Space Requirements

For a dataset of N images:
- **Stage 1 output**: ~5-10 MB per image (depends on num_crops)
- **Stage 2 output**: ~100 bytes per image (JSON metadata)
- **Stage 3 output**: ~0.5-2 MB per image (PNG, varies with size)

Example for 10,000 images with 15 crops:
- Preprocessed tensors: ~50-100 GB
- Metadata: ~1 MB
- Final crops: ~5-20 GB

## Models

Supported DINOv2/v3 models:
- `facebook/dinov2-small` - Fastest
- `facebook/dinov2-base` - Balanced (recommended)
- `facebook/dinov2-large` - Most accurate
- `facebook/dinov2-giant` - Best quality, slowest
- `facebook/dinov3-vits16-pretrain-lvd1689m` - DINOv3 variant

## Troubleshooting

**Out of Memory (Stage 2)**:
- Reduce `--batch_size`
- Use smaller model
- Reduce `--num_crops` in Stage 1

**Slow Stage 1**:
- Increase `--num_workers`
- Reduce `--resize_to`
- Reduce `--num_crops`

**Disk Space Issues**:
- Process dataset in chunks using `--start_idx` and `--max_samples`
- Clean up preprocessed tensors after Stage 3
- Use JPEG instead of PNG for final output

## File Structure

```
mj-data/
├── stage1_preprocess.py              # Stage 1: CPU preprocessing
├── stage2_inference.py                # Stage 2: GPU inference
├── stage3_save_crops.py               # Stage 3: Standard version
├── stage3_save_crops_optimized.py     # Stage 3: Optimized version
├── run_pipeline.py                    # Complete pipeline runner
├── smart_crop.py                      # Original single-process implementation
├── main.py                            # Original multi-GPU implementation
└── README_PIPELINE.md                 # This file
```

## Comparison with Original Implementation

| Feature | Original (main.py) | New Pipeline |
|---------|-------------------|--------------|
| Stages | Single stage | 3 separate stages |
| Preprocessing | On-the-fly | Pre-computed, saved |
| Resumable | No | Yes, stage-by-stage |
| CPU Usage | Limited | Full multiprocessing |
| GPU Usage | Multi-GPU (Accelerate) | Single/Multi-GPU (PyTorch) |
| Disk I/O | High (repeated reads) | Low (cached tensors) |
| Debugging | Difficult | Easy (stage isolation) |
| Flexibility | Low | High (swap models, params) |

## License

Same as parent project.
