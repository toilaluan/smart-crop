"""
Stage 1: CPU Multiprocessing - Preprocess Images and Save Tensors

This script loads images from a dataset, resizes them, generates crop candidates,
applies model preprocessing, and saves the tensors to disk for later inference.

Output format: .pt files containing:
    - image_id: str
    - tensors: Tensor [num_images, C, H, W] preprocessed for model
    - crop_boxes: List of crop boxes [(left, top, right, bottom), ...]
    - original_size: (width, height)
"""

import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Any


class ImagePreprocessor:
    """Preprocesses images and generates crop tensors."""

    def __init__(
        self,
        num_crops: int = 16,
        resize_to: int = 1024,
        model_input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.num_crops = num_crops
        self.resize_to = resize_to
        self.model_input_size = model_input_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor [C, H, W] in range [0, 1]."""
        return TF.to_tensor(image)

    def _resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize tensor to target size while maintaining aspect ratio."""
        _, height, width = tensor.shape

        if width >= height:
            new_height = self.resize_to
            new_width = int(width * (self.resize_to / height)) // 16 * 16
        else:
            new_width = self.resize_to
            new_height = int(height * (self.resize_to / width)) // 16 * 16

        if new_width != width or new_height != height:
            tensor = TF.resize(tensor, [new_height, new_width], antialias=True)

        return tensor

    def _generate_crop_tensors(
        self, tensor: torch.Tensor, crop_size: int, num_crops: int
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """Generate N evenly-spaced crops along the longer dimension as tensors."""
        _, height, width = tensor.shape
        crops = []
        boxes = []

        if width > height:
            # Crop along width
            max_left = width - crop_size
            if num_crops == 1:
                positions = [max_left // 2]
            else:
                positions = [
                    int(i * max_left / (num_crops - 1)) for i in range(num_crops)
                ]

            for left in positions:
                box = (left, 0, left + crop_size, crop_size)
                cropped = tensor[:, 0:crop_size, left:left + crop_size]
                crops.append(cropped)
                boxes.append(box)
        else:
            # Crop along height
            max_top = height - crop_size
            if num_crops == 1:
                positions = [max_top // 2]
            else:
                positions = [
                    int(i * max_top / (num_crops - 1)) for i in range(num_crops)
                ]

            for top in positions:
                box = (0, top, crop_size, top + crop_size)
                cropped = tensor[:, top:top + crop_size, 0:crop_size]
                crops.append(cropped)
                boxes.append(box)

        return crops, boxes

    def _normalize_and_resize_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize to model input size and normalize."""
        # Resize to model input size
        tensor = TF.resize(
            tensor,
            [self.model_input_size, self.model_input_size],
            antialias=True
        )
        # Normalize
        tensor = (tensor - self.mean) / self.std
        return tensor

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single dataset item.

        Returns:
            dict with:
                - image_id: str
                - tensors: Tensor [num_images, C, H, W] preprocessed for model
                - crop_boxes: List of crop boxes
                - original_size: (width, height)
                - original_crops: List of original-sized crop tensors [C, H, W] (for saving later)
                - success: bool
                - error: str or None
        """
        image = item["image"]
        image_id = item["id"]

        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to tensor immediately
            tensor = self._pil_to_tensor(image)  # [C, H, W]

            # Resize image
            if self.resize_to is not None:
                tensor = self._resize_tensor(tensor)

            _, height, width = tensor.shape

            # If image is square, just return original
            if width == height:
                crop_boxes = [(0, 0, width, height)]
                original_crops = [tensor]
                # Prepare for model
                model_tensor = self._normalize_and_resize_for_model(tensor).unsqueeze(0)
            else:
                # Generate crops as tensors
                crop_size = min(width, height)
                crop_tensors, crop_boxes_list = self._generate_crop_tensors(
                    tensor, crop_size, self.num_crops
                )

                crop_boxes = [None] + crop_boxes_list  # None for original
                original_crops = [tensor] + crop_tensors  # Store original-sized crops

                # Prepare all tensors for model: [original, crop1, crop2, ...]
                all_tensors = [self._normalize_and_resize_for_model(tensor)]
                for crop_t in crop_tensors:
                    all_tensors.append(self._normalize_and_resize_for_model(crop_t))

                model_tensor = torch.stack(all_tensors, dim=0)  # [num_images, C, H, W]

            return {
                "image_id": image_id,
                "tensors": model_tensor,
                "crop_boxes": crop_boxes,
                "original_size": (width, height),
                "original_crops": original_crops,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "image_id": image_id,
                "tensors": None,
                "crop_boxes": None,
                "original_size": None,
                "success": False,
                "error": str(e),
            }


def process_worker(args):
    """Worker function for multiprocessing."""
    item, preprocessor_config = args
    preprocessor = ImagePreprocessor(**preprocessor_config)
    return preprocessor.process_item(item)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Preprocess images and save tensors"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ujin-song/pexels-image-60k",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed_tensors",
        help="Output directory for preprocessed tensors",
    )
    parser.add_argument(
        "--num_crops",
        type=int,
        default=15,
        help="Number of candidate crops to evaluate per image",
    )
    parser.add_argument(
        "--resize_to",
        type=int,
        default=1024,
        help="Resize shorter side to this value before cropping",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=224,
        help="Model input size (e.g., 224 for DINOv2/v3)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU workers (default: cpu_count)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index in dataset"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set number of workers
    num_workers = args.num_workers or cpu_count()

    print(f"Stage 1: Preprocessing Images")
    print(f"=" * 60)
    print(f"Dataset: {args.dataset} (split: {args.split})")
    print(f"Output directory: {output_dir}")
    print(f"Number of crops per image: {args.num_crops}")
    print(f"Resize to: {args.resize_to}")
    print(f"Model input size: {args.model_input_size}")
    print(f"CPU workers: {num_workers}")
    print()

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)

    # Select subset if specified
    if args.max_samples is not None:
        end_idx = min(args.start_idx + args.max_samples, len(ds))
        ds = ds.select(range(args.start_idx, end_idx))
        print(f"Processing samples {args.start_idx} to {end_idx} ({len(ds)} total)")
    else:
        if args.start_idx > 0:
            ds = ds.select(range(args.start_idx, len(ds)))
        print(f"Processing {len(ds)} samples")

    # Prepare preprocessor configuration
    preprocessor_config = {
        "num_crops": args.num_crops,
        "resize_to": args.resize_to,
        "model_input_size": args.model_input_size,
    }

    # Prepare arguments for workers
    worker_args = [(item, preprocessor_config) for item in ds]

    # Process with multiprocessing
    print("\nProcessing images...")
    success_count = 0
    error_count = 0

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_worker, worker_args),
            total=len(worker_args),
            desc="Preprocessing"
        ))

    # Save results
    print("\nSaving preprocessed tensors...")
    error_log = []

    for result in tqdm(results, desc="Saving"):
        if result["success"]:
            # Save tensor file
            tensor_file = output_dir / f"{result['image_id']}.pt"
            torch.save({
                "image_id": result["image_id"],
                "tensors": result["tensors"],
                "crop_boxes": result["crop_boxes"],
                "original_size": result["original_size"],
                "original_crops": result["original_crops"],
            }, tensor_file)
            success_count += 1
        else:
            error_log.append(f"{result['image_id']}\t{result['error']}")
            error_count += 1

    # Save error log
    if error_log:
        error_file = output_dir / "preprocessing_errors.txt"
        with open(error_file, "w") as f:
            f.write("image_id\terror\n")
            f.write("\n".join(error_log))
        print(f"\nErrors saved to: {error_file}")

    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total: {success_count + error_count}")
    print(f"Tensors saved to: {output_dir}")


if __name__ == "__main__":
    main()
