"""
Stage 3 (Optimized): Save Cropped Images

This optimized version reads preprocessed tensors from Stage 1 and metadata from Stage 2,
then saves the cropped images without re-loading the original dataset.

This is more efficient because:
1. We already have the crop tensors saved in Stage 1 preprocessed files
2. We just need to load the correct crop based on best_crop_idx from Stage 2
3. No need to reload the entire dataset

Input:
    - Preprocessed tensors from Stage 1 (.pt files)
    - Crop metadata from Stage 2 (crop_metadata.json)
Output:
    - Cropped images saved to disk
"""

import os
import argparse
import json
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict, Any


def save_crop_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function to load tensor and save the best crop.

    Args:
        args: (metadata_entry, tensor_dir, output_dir)

    Returns:
        Result dict with success status
    """
    metadata, tensor_dir, output_dir = args
    image_id = metadata["image_id"]
    best_crop_idx = metadata["best_crop_idx"]

    try:
        # Load preprocessed tensor file
        tensor_file = Path(tensor_dir) / f"{image_id}.pt"
        if not tensor_file.exists():
            return {
                "image_id": image_id,
                "success": False,
                "error": f"Tensor file not found: {tensor_file}",
            }

        data = torch.load(tensor_file, map_location="cpu")

        # Get the original-sized crops (saved by Stage 1)
        # original_crops[0] is the full image
        # original_crops[1:] are the crop candidates
        original_crops = data["original_crops"]

        # best_crop_idx corresponds to the position in the list
        # (0 = original full image, 1+ = crops)
        best_crop_tensor = original_crops[best_crop_idx]

        # Convert tensor to PIL Image
        # Tensor is [C, H, W] in range [0, 1]
        pil_image = TF.to_pil_image(best_crop_tensor)

        # Save image
        output_path = Path(output_dir) / f"{image_id}.png"
        pil_image.save(output_path)

        return {
            "image_id": image_id,
            "success": True,
            "output_path": str(output_path),
            "error": None,
        }

    except Exception as e:
        return {
            "image_id": image_id,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 (Optimized): Save cropped images from preprocessed tensors"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="./crop_metadata.json",
        help="JSON file containing crop metadata from Stage 2",
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default="./preprocessed_tensors",
        help="Directory containing preprocessed tensors from Stage 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cropped_images",
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU workers (default: cpu_count)",
    )

    args = parser.parse_args()

    metadata_file = Path(args.metadata_file)
    tensor_dir = Path(args.tensor_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set number of workers
    num_workers = args.num_workers or cpu_count()

    print(f"Stage 3 (Optimized): Save Cropped Images")
    print(f"=" * 60)
    print(f"Metadata file: {metadata_file}")
    print(f"Tensor directory: {tensor_dir}")
    print(f"Output directory: {output_dir}")
    print(f"CPU workers: {num_workers}")
    print()

    # Load metadata
    print("Loading metadata...")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata_list = json.load(f)

    print(f"Loaded metadata for {len(metadata_list)} images")

    # Check tensor directory exists
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")

    # Prepare worker arguments
    worker_args = [
        (metadata, tensor_dir, output_dir)
        for metadata in metadata_list
    ]

    # Process with multiprocessing
    print("\nSaving cropped images...")
    success_count = 0
    error_count = 0
    error_log = []

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(save_crop_worker, worker_args),
            total=len(worker_args),
            desc="Processing"
        ))

    # Collect results
    for result in results:
        if result["success"]:
            success_count += 1
        else:
            error_count += 1
            error_log.append(f"{result['image_id']}\t{result['error']}")

    # Save error log if there were errors
    if error_log:
        error_file = output_dir / "cropping_errors.txt"
        with open(error_file, "w") as f:
            f.write("image_id\terror\n")
            f.write("\n".join(error_log))
        print(f"\nErrors saved to: {error_file}")

    # Save processing summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Stage 3 (Optimized) Summary\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {error_count}\n")
        f.write(f"Total: {success_count + error_count}\n")

    print("\n" + "=" * 60)
    print("Stage 3 Complete!")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total: {success_count + error_count}")
    print(f"Cropped images saved to: {output_dir}")


if __name__ == "__main__":
    main()
