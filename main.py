"""
Smart Crop Large Dataset with Multi-GPU Support using Accelerate
"""

import os
from datasets import load_dataset
from accelerate import Accelerator
from smart_crop import SmartImageCropper
from tqdm import tqdm
import argparse


def process_batch(batch, cropper, num_crops=16, resize_to=1024):
    """
    Process a batch of images through smart cropping.

    Args:
        batch: Batch from dataset containing 'image' and 'id' fields
        cropper: SmartImageCropper instance
        num_crops: Number of candidate crops to evaluate
        resize_to: Target size for resizing before cropping

    Returns:
        List of dictionaries with results
    """
    results = []

    for idx in range(len(batch['id'])):
        image = batch['image'][idx]
        image_id = batch['id'][idx]

        try:
            # Perform smart crop
            best_crop, crop_box, similarity = cropper.crop(
                image,
                num_crops=num_crops,
                resize_to=resize_to
            )

            results.append({
                'id': image_id,
                'crop_box': crop_box,
                'similarity': similarity,
                'cropped_image': best_crop,
                'success': True,
                'error': None
            })
        except Exception as e:
            results.append({
                'id': image_id,
                'crop_box': None,
                'similarity': None,
                'cropped_image': None,
                'success': False,
                'error': str(e)
            })

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
        if result['success'] and result['cropped_image'] is not None:
            # Save image
            image_filename = f"{result['id']}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            result['cropped_image'].save(image_path, quality=95)

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
    parser = argparse.ArgumentParser(description='Smart crop large datasets with multi-GPU support')
    parser.add_argument('--dataset', type=str, default='ujin-song/pexels-image-60k',
                        help='Dataset name or path')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to process')
    parser.add_argument('--output_dir', type=str, default='./cropped_output',
                        help='Output directory for cropped images')
    parser.add_argument('--model_name', type=str,
                        default='facebook/dinov2-base',
                        help='DINOv2 model name')
    parser.add_argument('--num_crops', type=int, default=16,
                        help='Number of candidate crops to evaluate per image')
    parser.add_argument('--resize_to', type=int, default=1024,
                        help='Resize shorter side to this value before cropping')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in dataset')

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Create output directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * accelerator.num_processes}")

    # Wait for main process to create directory
    accelerator.wait_for_everyone()

    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset: {args.dataset} (split: {args.split})")

    ds = load_dataset(args.dataset, split=args.split)

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

    cropper = SmartImageCropper(
        model_name=args.model_name,
        device=accelerator.device
    )

    # Prepare model with accelerator
    cropper.model = accelerator.prepare(cropper.model)

    if accelerator.is_main_process:
        print(f"Model loaded on {accelerator.num_processes} GPU(s)")
        print("Starting processing...")

    # Split dataset across processes
    # Each process will handle a portion of the dataset
    with accelerator.split_between_processes(list(range(len(ds)))) as subset_indices:
        # Open metadata file for this process
        metadata_path = os.path.join(
            args.output_dir,
            f'metadata_process_{accelerator.process_index}.txt'
        )

        with open(metadata_path, 'w') as metadata_file:
            # Write header
            metadata_file.write("id\tfilename\tcrop_box\tsimilarity\n")

            # Process images in batches
            subset_size = len(subset_indices)

            # Create progress bar (only show on main process)
            if accelerator.is_main_process:
                pbar = tqdm(total=len(ds), desc="Processing images")

            for i in range(0, subset_size, args.batch_size):
                batch_indices = subset_indices[i:i + args.batch_size]
                batch_ds = ds.select(batch_indices)

                # Process batch
                results = process_batch(
                    batch_ds,
                    cropper,
                    num_crops=args.num_crops,
                    resize_to=args.resize_to
                )

                # Save results
                save_results(results, args.output_dir, metadata_file)

                # Update progress bar on main process
                if accelerator.is_main_process:
                    pbar.update(len(batch_indices) * accelerator.num_processes)

            if accelerator.is_main_process:
                pbar.close()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Merge metadata files on main process
    if accelerator.is_main_process:
        print("\nMerging metadata files...")
        final_metadata_path = os.path.join(args.output_dir, 'metadata.txt')

        with open(final_metadata_path, 'w') as final_file:
            final_file.write("id\tfilename\tcrop_box\tsimilarity\n")

            for proc_idx in range(accelerator.num_processes):
                proc_metadata_path = os.path.join(
                    args.output_dir,
                    f'metadata_process_{proc_idx}.txt'
                )

                if os.path.exists(proc_metadata_path):
                    with open(proc_metadata_path, 'r') as proc_file:
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
        with open(final_metadata_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                if 'ERROR' in line:
                    error_count += 1
                else:
                    success_count += 1

        print("\nResults:")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {error_count}")
        print(f"  Total: {success_count + error_count}")


if __name__ == "__main__":
    main()
