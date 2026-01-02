"""
Smart Crop Large Dataset with Multi-GPU Support using Accelerate
"""

import os
from datasets import load_dataset
from accelerate import Accelerator
from smart_crop import SmartImageCropper
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Dict, Any
import numpy as np


class CropDataset(Dataset):
    """
    Dataset that generates preprocessed tensors for original image + all crop candidates.
    All preprocessing is done on tensors for maximum efficiency.
    """

    def __init__(
        self,
        hf_dataset,
        num_crops: int = 16,
        resize_to: int = 1024,
        model_input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset with 'image' and 'id' fields
            num_crops: Number of candidate crops to generate per image
            resize_to: Target size for resizing before cropping
            model_input_size: Model input size (e.g., 224 for DINOv2)
            mean: Normalization mean for ImageNet
            std: Normalization std for ImageNet
        """
        self.hf_dataset = hf_dataset
        self.num_crops = num_crops
        self.resize_to = resize_to
        self.model_input_size = model_input_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.hf_dataset)

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
                # Tensor crop: [C, H, W] -> crop uses [top:bottom, left:right]
                cropped = tensor[:, 0:crop_size, left : left + crop_size]
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
                # Tensor crop: [C, H, W] -> crop uses [top:bottom, left:right]
                cropped = tensor[:, top : top + crop_size, 0:crop_size]
                crops.append(cropped)
                boxes.append(box)

        return crops, boxes

    def _normalize_and_resize_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize to model input size and normalize."""
        # Resize to model input size
        tensor = TF.resize(
            tensor, [self.model_input_size, self.model_input_size], antialias=True
        )
        # Normalize
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx):
        """
        Returns preprocessed tensors ready for model inference.

        Returns:
            dict with:
                - image_id: str
                - tensors: Tensor of shape [num_images, C, H, W] (preprocessed for model)
                - crop_boxes: List of crop boxes
                - original_size: Tuple of (width, height)
                - original_crops: List of original crop tensors (for saving later)
        """
        item = self.hf_dataset[idx]
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
                model_tensor = self._normalize_and_resize_for_model(tensor).unsqueeze(
                    0
                )  # [1, C, H, W]
            else:
                # Generate crops as tensors
                crop_size = min(width, height)
                crop_tensors, crop_boxes_list = self._generate_crop_tensors(
                    tensor, crop_size, self.num_crops
                )

                # Store original-sized crops for saving later
                original_crops = [tensor] + crop_tensors
                crop_boxes = [None] + crop_boxes_list

                # Prepare all tensors for model: [original, crop1, crop2, ...]
                all_tensors = []
                for t in original_crops:
                    all_tensors.append(self._normalize_and_resize_for_model(t))

                model_tensor = torch.stack(all_tensors, dim=0)  # [num_images, C, H, W]

            return {
                "image_id": image_id,
                "tensors": model_tensor,  # Preprocessed for model
                "crop_boxes": crop_boxes,
                "original_size": (width, height),
                "original_crops": original_crops,  # For saving later
                "success": True,
                "error": None,
            }

        except Exception as e:
            # Return error case
            return {
                "image_id": image_id,
                "tensors": None,
                "crop_boxes": None,
                "original_size": None,
                "original_crops": None,
                "success": False,
                "error": str(e),
            }


def collate_fn(batch):
    """
    Custom collate function that concatenates preprocessed tensors.

    Batch contains items, each with preprocessed tensors ready for model.
    We concatenate them into a single batch tensor.

    Returns:
        dict with:
            - model_inputs: Tensor of shape [total_images, C, H, W] ready for model
            - metadata: List of dicts with reconstruction info
    """
    all_tensors = []
    metadata = []

    for item in batch:
        if not item["success"]:
            # Keep error items for later processing
            metadata.append(
                {
                    "image_id": item["image_id"],
                    "success": False,
                    "error": item["error"],
                    "num_images": 0,
                    "start_idx": 0,
                    "crop_boxes": None,
                    "original_size": None,
                    "original_crops": None,
                }
            )
            continue

        num_images = item["tensors"].shape[0]
        start_idx = sum(m["num_images"] for m in metadata if m["success"])

        # Add preprocessed tensors to list
        all_tensors.append(item["tensors"])

        # Store metadata for reconstruction
        metadata.append(
            {
                "image_id": item["image_id"],
                "success": True,
                "error": None,
                "num_images": num_images,
                "start_idx": start_idx,
                "crop_boxes": item["crop_boxes"],
                "original_size": item["original_size"],
                "original_crops": item["original_crops"],
            }
        )

    # Concatenate all tensors into single batch
    if len(all_tensors) > 0:
        model_inputs = torch.cat(all_tensors, dim=0)  # [total_images, C, H, W]
    else:
        model_inputs = None

    return {"model_inputs": model_inputs, "metadata": metadata}


def process_batch(batch, cropper, device):
    """
    Process a batch of preprocessed tensors through smart cropping.

    Args:
        batch: Batch from DataLoader with preprocessed tensors
        cropper: SmartImageCropper instance
        device: Device to move tensors to

    Returns:
        List of dictionaries with results
    """
    model_inputs = batch["model_inputs"]
    metadata = batch["metadata"]
    results = []

    if model_inputs is None or len(metadata) == 0:
        # Only errors in batch
        for meta in metadata:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": None,
                    "similarity": None,
                    "cropped_tensor": None,
                    "success": False,
                    "error": meta.get("error", "Unknown error"),
                }
            )
        return results

    # Move tensors to device and compute features directly (skip processor)
    model_inputs = model_inputs.to(device)

    with torch.inference_mode():
        # Direct model inference on preprocessed tensors
        outputs = cropper.model(pixel_values=model_inputs)
        all_features = outputs.pooler_output  # [total_images, feature_dim]

    # Process each item in the batch
    for meta in metadata:
        if not meta["success"]:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": None,
                    "similarity": None,
                    "cropped_tensor": None,
                    "success": False,
                    "error": meta["error"],
                }
            )
            continue

        start_idx = meta["start_idx"]
        num_images = meta["num_images"]
        crop_boxes = meta["crop_boxes"]
        original_crops = meta["original_crops"]

        # Extract features for this item
        item_features = all_features[start_idx : start_idx + num_images]

        # If image was square, just return it
        if num_images == 1:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": crop_boxes[0],
                    "similarity": 1.0,
                    "cropped_tensor": original_crops[0],
                    "success": True,
                    "error": None,
                }
            )
            continue

        # First feature is original, rest are crops
        original_feature = item_features[0:1]
        crop_features = item_features[1:]

        # Compute similarities
        similarities = cropper._cosine_similarity_batch(original_feature, crop_features)

        # Find best crop
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        best_box = crop_boxes[best_idx + 1]  # +1 because crop_boxes[0] is None
        best_crop_tensor = original_crops[1 + best_idx]  # +1 to skip original

        results.append(
            {
                "id": meta["image_id"],
                "crop_box": best_box,
                "similarity": best_similarity,
                "cropped_tensor": best_crop_tensor,
                "success": True,
                "error": None,
            }
        )

    return results


def save_results(results, output_dir, metadata_file):
    """
    Save cropped images to disk and write metadata.

    Args:
        results: List of result dictionaries with tensors
        output_dir: Directory to save images
        metadata_file: File handle for writing metadata
    """
    for result in results:
        if result["success"] and result["cropped_tensor"] is not None:
            # Convert tensor to PIL Image for saving
            # Tensor is [C, H, W] in range [0, 1]
            tensor = result["cropped_tensor"]
            pil_image = TF.to_pil_image(tensor)

            # Save image
            image_filename = f"{result['id']}.png"
            image_path = os.path.join(output_dir, image_filename)
            pil_image.save(image_path)

            # Write metadata
            metadata_line = (
                f"{result['id']}\t{image_filename}\t"
                f"{result['crop_box']}\t{result['similarity']:.4f}\n"
            )
            metadata_file.write(metadata_line)
        else:
            # Log error
            error_line = f"{result['id']}\tERROR\t{result['error']}\n"
            metadata_file.write(error_line)


def main():
    parser = argparse.ArgumentParser(
        description="Smart crop large datasets with multi-GPU support"
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
        default="./cropped_output",
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="DINOv3 model name",
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
        "--batch_size",
        type=int,
        default=4,
        help="Number of images per batch (effective batch will be batch_size * (num_crops + 1))",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for preprocessing",
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

    # Initialize accelerator
    accelerator = Accelerator()

    # Create output directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Images per batch: {args.batch_size}")
        print(f"Number of crops per image: {args.num_crops}")
        print(
            f"Effective batch size (total images): ~{args.batch_size * (args.num_crops + 1)}"
        )
        print(f"DataLoader workers: {args.num_workers}")

    # Wait for main process to create directory
    accelerator.wait_for_everyone()

    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset: {args.dataset} (split: {args.split})")

    ds = load_dataset(
        args.dataset,
        split=args.split,
        num_proc=16,
        data_files="data/train-0000*-of-00171.parquet",
    )

    # Select subset if specified
    if args.max_samples is not None:
        end_idx = min(args.start_idx + args.max_samples, len(ds))
        ds = ds.select(range(args.start_idx, end_idx))
        if accelerator.is_main_process:
            print(f"Processing samples {args.start_idx} to {end_idx} ({len(ds)} total)")
    else:
        if args.start_idx > 0:
            ds = ds.select(range(args.start_idx, len(ds)))
        if accelerator.is_main_process:
            print(f"Processing {len(ds)} samples")

    # Initialize cropper on each GPU
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_name}")

    cropper = SmartImageCropper(model_name=args.model_name, device=accelerator.device)

    # Prepare model with accelerator
    cropper.model = accelerator.prepare(cropper.model)

    if accelerator.is_main_process:
        print(f"Model loaded on {accelerator.num_processes} GPU(s)")
        print("Starting processing...")

    # Split dataset across processes
    with accelerator.split_between_processes(list(range(len(ds)))) as subset_indices:
        # Create subset dataset for this process
        subset_ds = ds.select(subset_indices)

        # Create CropDataset wrapper
        crop_dataset = CropDataset(
            hf_dataset=subset_ds,
            num_crops=args.num_crops,
            resize_to=args.resize_to,
            model_input_size=cropper.processor.size["height"],  # DINOv2 uses 224
        )

        # Create DataLoader with custom collate function
        dataloader = DataLoader(
            crop_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # Open metadata file for this process
        metadata_path = os.path.join(
            args.output_dir, f"metadata_process_{accelerator.process_index}.txt"
        )

        with open(metadata_path, "w") as metadata_file:
            # Write header
            metadata_file.write("id\tfilename\tcrop_box\tsimilarity\n")

            # Create progress bar (only show on main process)
            if accelerator.is_main_process:
                pbar = tqdm(total=len(ds), desc="Processing images")

            # Process batches
            for batch in dataloader:
                # Process batch with batched inference
                results = process_batch(batch, cropper, accelerator.device)

                # Save results
                save_results(results, args.output_dir, metadata_file)

                # Update progress bar on main process
                if accelerator.is_main_process:
                    pbar.update(len(results) * accelerator.num_processes)

            if accelerator.is_main_process:
                pbar.close()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Merge metadata files on main process
    if accelerator.is_main_process:
        print("\nMerging metadata files...")
        final_metadata_path = os.path.join(args.output_dir, "metadata.txt")

        with open(final_metadata_path, "w") as final_file:
            final_file.write("id\tfilename\tcrop_box\tsimilarity\n")

            for proc_idx in range(accelerator.num_processes):
                proc_metadata_path = os.path.join(
                    args.output_dir, f"metadata_process_{proc_idx}.txt"
                )

                if os.path.exists(proc_metadata_path):
                    with open(proc_metadata_path, "r") as proc_file:
                        # Skip header
                        next(proc_file)
                        # Copy content
                        final_file.write(proc_file.read())

                    # Remove process-specific file
                    os.remove(proc_metadata_path)

        print("Processing complete!")
        print(f"Cropped images saved to: {args.output_dir}")
        print(f"Metadata saved to: {final_metadata_path}")

        # Count successful and failed images
        success_count = 0
        error_count = 0
        with open(final_metadata_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                if "ERROR" in line:
                    error_count += 1
                else:
                    success_count += 1

        print("\nResults:")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {error_count}")
        print(f"  Total: {success_count + error_count}")


if __name__ == "__main__":
    main()
