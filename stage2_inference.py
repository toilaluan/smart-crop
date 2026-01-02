"""
Stage 2: GPU Inference - Identify Best Crops

This script loads preprocessed tensors from Stage 1, runs inference to identify
the best crop for each image, and saves metadata to disk.

Input: .pt files from Stage 1
Output: metadata.json containing crop decisions
"""

import os
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    """Dataset that loads preprocessed tensors from disk."""

    def __init__(self, tensor_dir: Path):
        self.tensor_dir = tensor_dir
        self.tensor_files = sorted(list(tensor_dir.glob("*.pt")))

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_file = self.tensor_files[idx]
        data = torch.load(tensor_file, map_location="cpu")
        return data


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that batches preprocessed tensors.

    Returns:
        dict with:
            - model_inputs: Tensor [total_images, C, H, W] ready for model
            - metadata: List of dicts with reconstruction info
    """
    all_tensors = []
    metadata = []

    for item in batch:
        num_images = item["tensors"].shape[0]
        start_idx = sum(m["num_images"] for m in metadata)

        all_tensors.append(item["tensors"])

        metadata.append({
            "image_id": item["image_id"],
            "num_images": num_images,
            "start_idx": start_idx,
            "crop_boxes": item["crop_boxes"],
            "original_size": item["original_size"],
        })

    # Concatenate all tensors into single batch
    model_inputs = torch.cat(all_tensors, dim=0)  # [total_images, C, H, W]

    return {"model_inputs": model_inputs, "metadata": metadata}


def compute_similarities(
    original_feature: torch.Tensor,
    crop_features: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between original and crop features.

    Args:
        original_feature: [1, feature_dim]
        crop_features: [num_crops, feature_dim]

    Returns:
        Similarity scores [num_crops]
    """
    original_norm = original_feature / original_feature.norm(dim=-1, keepdim=True)
    crop_norm = crop_features / crop_features.norm(dim=-1, keepdim=True)
    similarities = (crop_norm * original_norm).sum(dim=-1)
    return similarities


def process_batch(
    batch: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Process a batch of tensors and identify best crops.

    Returns:
        List of metadata entries with crop decisions
    """
    model_inputs = batch["model_inputs"].to(device)
    metadata = batch["metadata"]
    results = []

    with torch.inference_mode():
        outputs = model(pixel_values=model_inputs)
        all_features = outputs.pooler_output  # [total_images, feature_dim]

    # Process each item in batch
    for meta in metadata:
        start_idx = meta["start_idx"]
        num_images = meta["num_images"]
        crop_boxes = meta["crop_boxes"]

        # Extract features for this item
        item_features = all_features[start_idx:start_idx + num_images]

        # If image was square (only 1 image)
        if num_images == 1:
            results.append({
                "image_id": meta["image_id"],
                "crop_box": crop_boxes[0],
                "similarity": 1.0,
                "best_crop_idx": 0,
                "original_size": meta["original_size"],
            })
            continue

        # First feature is original, rest are crops
        original_feature = item_features[0:1]
        crop_features = item_features[1:]

        # Compute similarities
        similarities = compute_similarities(original_feature, crop_features)

        # Find best crop
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        best_box = crop_boxes[best_idx + 1]  # +1 because crop_boxes[0] is None

        results.append({
            "image_id": meta["image_id"],
            "crop_box": best_box,
            "similarity": float(best_similarity),
            "best_crop_idx": best_idx + 1,  # +1 to account for original at idx 0
            "original_size": meta["original_size"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: GPU inference to identify best crops"
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default="./preprocessed_tensors",
        help="Directory containing preprocessed tensors from Stage 1",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./crop_metadata.json",
        help="Output JSON file for crop metadata",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Model name for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images per batch (will be multiplied by num_crops)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    args = parser.parse_args()

    tensor_dir = Path(args.tensor_dir)
    output_file = Path(args.output_file)

    print(f"Stage 2: GPU Inference")
    print(f"=" * 60)
    print(f"Tensor directory: {tensor_dir}")
    print(f"Output file: {output_file}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Check if tensor directory exists
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")

    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = TensorDataset(tensor_dir)
    print(f"Found {len(dataset)} preprocessed tensor files")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Process all batches
    print("\nRunning inference...")
    all_results = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        results = process_batch(batch, model, device)
        all_results.extend(results)

    # Save metadata
    print(f"\nSaving metadata to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Stage 2 Complete!")
    print(f"Processed {len(all_results)} images")
    print(f"Metadata saved to: {output_file}")

    # Print statistics
    similarities = [r["similarity"] for r in all_results]
    if similarities:
        print(f"\nSimilarity Statistics:")
        print(f"  Mean: {sum(similarities) / len(similarities):.4f}")
        print(f"  Min: {min(similarities):.4f}")
        print(f"  Max: {max(similarities):.4f}")


if __name__ == "__main__":
    main()
