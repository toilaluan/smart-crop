"""
Smart Crop Large Dataset with Multi-GPU Support using Accelerate
"""

import os
from datasets import load_dataset
from accelerate import Accelerator
from smart_crop import SmartImageCropper
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple


class CropDataset(Dataset):
    """
    Dataset that generates original image + all crop candidates for each image.
    Preprocessing is done in __getitem__ for efficiency.
    """

    def __init__(
        self,
        hf_dataset,
        num_crops: int = 16,
        resize_to: int = 1024,
        processor=None,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset with 'image' and 'id' fields
            num_crops: Number of candidate crops to generate per image
            resize_to: Target size for resizing before cropping
            processor: Image processor for the model
        """
        self.hf_dataset = hf_dataset
        self.num_crops = num_crops
        self.resize_to = resize_to
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size while maintaining aspect ratio."""
        width, height = image.size

        if width >= height:
            new_height = self.resize_to
            new_width = int(width * (self.resize_to / height)) // 16 * 16
        else:
            new_width = self.resize_to
            new_height = int(height * (self.resize_to / width)) // 16 * 16

        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.LANCZOS)

        return image

    def _generate_crops(
        self, image: Image.Image, crop_size: int, num_crops: int
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Generate N evenly-spaced crops along the longer dimension."""
        width, height = image.size
        crops = []

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
                cropped = image.crop(box)
                crops.append((cropped, box))
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
                cropped = image.crop(box)
                crops.append((cropped, box))

        return crops

    def __getitem__(self, idx):
        """
        Returns preprocessed tensors for original image + all crops.

        Returns:
            dict with:
                - image_id: str
                - images: List of PIL Images (original + crops)
                - crop_boxes: List of crop boxes (None for original, then actual boxes)
                - original_size: Tuple of (width, height)
        """
        item = self.hf_dataset[idx]
        image = item["image"]
        image_id = item["id"]

        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image
            if self.resize_to is not None:
                image = self._resize_image(image)

            width, height = image.size

            # If image is square, just return original
            if width == height:
                images = [image]
                crop_boxes = [(0, 0, width, height)]
            else:
                # Generate crops
                crop_size = min(width, height)
                crops = self._generate_crops(image, crop_size, self.num_crops)

                # Prepare list: [original, crop1, crop2, ...]
                images = [image] + [crop_img for crop_img, _ in crops]
                crop_boxes = [None] + [box for _, box in crops]

            return {
                "image_id": image_id,
                "images": images,
                "crop_boxes": crop_boxes,
                "original_size": (width, height),
                "success": True,
                "error": None,
            }

        except Exception as e:
            # Return error case
            return {
                "image_id": image_id,
                "images": None,
                "crop_boxes": None,
                "original_size": None,
                "success": False,
                "error": str(e),
            }


def collate_fn(batch):
    """
    Custom collate function that flattens all images from the batch.

    Batch contains items, each with (1 original + n_crops) images.
    We flatten to process all images in one inference pass.

    Returns:
        dict with:
            - all_images: List of all PIL images to process
            - metadata: List of dicts with reconstruction info
    """
    all_images = []
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
                    "start_idx": len(all_images),
                }
            )
            continue

        num_images = len(item["images"])
        start_idx = len(all_images)

        # Add all images (original + crops) to the flat list
        all_images.extend(item["images"])

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
            }
        )

    return {"all_images": all_images, "metadata": metadata}


def process_batch(batch, cropper):
    """
    Process a batch of images through smart cropping using batched inference.

    Args:
        batch: Batch from DataLoader with flattened images
        cropper: SmartImageCropper instance

    Returns:
        List of dictionaries with results
    """
    all_images = batch["all_images"]
    metadata = batch["metadata"]
    results = []

    if len(all_images) == 0:
        # Only errors in batch
        for meta in metadata:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": None,
                    "similarity": None,
                    "cropped_image": None,
                    "success": False,
                    "error": meta.get("error", "Unknown error"),
                }
            )
        return results

    # Compute features for all images at once
    all_features = cropper._compute_features_batch(all_images)

    # Process each item in the batch
    for meta in metadata:
        if not meta["success"]:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": None,
                    "similarity": None,
                    "cropped_image": None,
                    "success": False,
                    "error": meta["error"],
                }
            )
            continue

        start_idx = meta["start_idx"]
        num_images = meta["num_images"]
        crop_boxes = meta["crop_boxes"]

        # Extract features for this item
        item_features = all_features[start_idx : start_idx + num_images]

        # If image was square, just return it
        if num_images == 1:
            results.append(
                {
                    "id": meta["image_id"],
                    "crop_box": crop_boxes[0],
                    "similarity": 1.0,
                    "cropped_image": all_images[start_idx],
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
        best_crop = all_images[start_idx + 1 + best_idx]  # +1 to skip original

        results.append(
            {
                "id": meta["image_id"],
                "crop_box": best_box,
                "similarity": best_similarity,
                "cropped_image": best_crop,
                "success": True,
                "error": None,
            }
        )

    return results


def save_results(results, output_dir, metadata_file):
    """
    Save cropped images to disk and write metadata.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save images
        metadata_file: File handle for writing metadata
    """
    for result in results:
        if result["success"] and result["cropped_image"] is not None:
            # Save image
            image_filename = f"{result['id']}.png"
            image_path = os.path.join(output_dir, image_filename)
            result["cropped_image"].save(image_path, quality=100)

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

    ds = load_dataset(args.dataset, split=args.split, num_proc=16)

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
            processor=cropper.processor,
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
                results = process_batch(batch, cropper)

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
