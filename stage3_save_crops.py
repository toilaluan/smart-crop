"""
Stage 3: Save Cropped Images

This script reads the metadata from Stage 2, loads the original images,
crops them according to the identified best crop boxes, and saves the final images.

Input: crop_metadata.json from Stage 2
Output: Cropped images saved to disk
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict, Any, Optional


def crop_and_save_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function to crop and save a single image.

    Args:
        args: (metadata_entry, dataset, output_dir)

    Returns:
        Result dict with success status
    """
    metadata, dataset_info, output_dir = args
    image_id = metadata["image_id"]
    crop_box = metadata["crop_box"]

    try:
        # Load image from dataset
        # Find the item by ID
        item = None
        for ds_item in dataset_info["items"]:
            if ds_item["id"] == image_id:
                item = ds_item
                break

        if item is None:
            return {
                "image_id": image_id,
                "success": False,
                "error": f"Image not found in dataset: {image_id}",
            }

        image = item["image"]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply crop if needed (crop_box can be None for square images)
        if crop_box is not None and crop_box != (0, 0, image.width, image.height):
            # crop_box format: (left, top, right, bottom)
            image = image.crop(crop_box)

        # Save cropped image
        output_path = Path(output_dir) / f"{image_id}.png"
        image.save(output_path)

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
        description="Stage 3: Crop and save images based on metadata"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="./crop_metadata.json",
        help="JSON file containing crop metadata from Stage 2",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ujin-song/pexels-image-60k",
        help="Dataset name or path (must match Stage 1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (must match Stage 1)",
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
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index in dataset (must match Stage 1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (must match Stage 1)",
    )

    args = parser.parse_args()

    metadata_file = Path(args.metadata_file)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set number of workers
    num_workers = args.num_workers or cpu_count()

    print(f"Stage 3: Crop and Save Images")
    print(f"=" * 60)
    print(f"Metadata file: {metadata_file}")
    print(f"Dataset: {args.dataset} (split: {args.split})")
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

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset(args.dataset, split=args.split)

    # Select subset if specified (must match Stage 1)
    if args.max_samples is not None:
        end_idx = min(args.start_idx + args.max_samples, len(ds))
        ds = ds.select(range(args.start_idx, end_idx))
        print(f"Using samples {args.start_idx} to {end_idx} ({len(ds)} total)")
    else:
        if args.start_idx > 0:
            ds = ds.select(range(args.start_idx, len(ds)))
        print(f"Using {len(ds)} samples")

    # Convert dataset to list for multiprocessing
    # (HuggingFace datasets don't pickle well)
    print("\nPreparing dataset items...")
    dataset_items = []
    for item in tqdm(ds, desc="Loading dataset items"):
        dataset_items.append({
            "id": item["id"],
            "image": item["image"],
        })

    dataset_info = {"items": dataset_items}

    # Prepare worker arguments
    worker_args = [
        (metadata, dataset_info, output_dir)
        for metadata in metadata_list
    ]

    # Process with multiprocessing
    print("\nCropping and saving images...")
    success_count = 0
    error_count = 0
    error_log = []

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(crop_and_save_worker, worker_args),
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
        f.write(f"Stage 3 Summary\n")
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
