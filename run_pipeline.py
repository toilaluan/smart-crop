"""
Complete Pipeline Runner

This script runs all three stages of the smart cropping pipeline sequentially:
1. Stage 1: CPU multiprocessing to preprocess images and save tensors
2. Stage 2: GPU inference to identify best crops
3. Stage 3: Save cropped images from tensors

This is a convenience script to run the entire pipeline with a single command.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        command: Command to run as list
        description: Description of the stage

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Running: {' '.join(command)}\n")

    try:
        result = subprocess.run(command, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete smart cropping pipeline"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="ujin-song/pexels-image-60k",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to process"
    )

    # Processing parameters
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
        "--model_name",
        type=str,
        default="facebook/dinov2-base",
        help="Model name for inference",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=224,
        help="Model input size (e.g., 224 for DINOv2/v3)",
    )

    # Resource parameters
    parser.add_argument(
        "--cpu_workers_stage1",
        type=int,
        default=None,
        help="Number of CPU workers for Stage 1 (default: cpu_count)",
    )
    parser.add_argument(
        "--cpu_workers_stage3",
        type=int,
        default=None,
        help="Number of CPU workers for Stage 3 (default: cpu_count)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU inference in Stage 2",
    )
    parser.add_argument(
        "--num_workers_stage2",
        type=int,
        default=4,
        help="DataLoader workers for Stage 2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Stage 2 inference",
    )

    # Data range parameters
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index in dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )

    # Output parameters
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default="./preprocessed_tensors",
        help="Directory for preprocessed tensors (Stage 1 output)",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="./crop_metadata.json",
        help="Metadata file (Stage 2 output)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cropped_images",
        help="Final output directory for cropped images",
    )

    # Pipeline control
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip Stage 1 (preprocessing)",
    )
    parser.add_argument(
        "--skip_stage2",
        action="store_true",
        help="Skip Stage 2 (inference)",
    )
    parser.add_argument(
        "--skip_stage3",
        action="store_true",
        help="Skip Stage 3 (saving crops)",
    )
    parser.add_argument(
        "--use_optimized_stage3",
        action="store_true",
        help="Use optimized Stage 3 (reads from tensors instead of dataset)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SMART CROPPING PIPELINE")
    print("=" * 80)
    print(f"Dataset: {args.dataset} (split: {args.split})")
    print(f"Model: {args.model_name}")
    print(f"Number of crops: {args.num_crops}")
    print(f"Resize to: {args.resize_to}")
    print(f"Model input size: {args.model_input_size}")
    print(f"Batch size: {args.batch_size}")
    if args.max_samples:
        print(f"Processing samples {args.start_idx} to {args.start_idx + args.max_samples}")
    print()

    success = True

    # Stage 1: Preprocessing
    if not args.skip_stage1:
        stage1_cmd = [
            sys.executable, "stage1_preprocess.py",
            "--dataset", args.dataset,
            "--split", args.split,
            "--output_dir", args.tensor_dir,
            "--num_crops", str(args.num_crops),
            "--resize_to", str(args.resize_to),
            "--model_input_size", str(args.model_input_size),
            "--start_idx", str(args.start_idx),
        ]

        if args.max_samples:
            stage1_cmd.extend(["--max_samples", str(args.max_samples)])
        if args.cpu_workers_stage1:
            stage1_cmd.extend(["--num_workers", str(args.cpu_workers_stage1)])

        success = run_command(stage1_cmd, "Stage 1: Preprocessing")
        if not success:
            return 1
    else:
        print("\n⊘ Skipping Stage 1 (preprocessing)")

    # Stage 2: Inference
    if not args.skip_stage2:
        stage2_cmd = [
            sys.executable, "stage2_inference.py",
            "--tensor_dir", args.tensor_dir,
            "--output_file", args.metadata_file,
            "--model_name", args.model_name,
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--num_workers", str(args.num_workers_stage2),
        ]

        success = run_command(stage2_cmd, "Stage 2: GPU Inference")
        if not success:
            return 1
    else:
        print("\n⊘ Skipping Stage 2 (inference)")

    # Stage 3: Save crops
    if not args.skip_stage3:
        if args.use_optimized_stage3:
            stage3_cmd = [
                sys.executable, "stage3_save_crops_optimized.py",
                "--metadata_file", args.metadata_file,
                "--tensor_dir", args.tensor_dir,
                "--output_dir", args.output_dir,
            ]
            if args.cpu_workers_stage3:
                stage3_cmd.extend(["--num_workers", str(args.cpu_workers_stage3)])

            success = run_command(stage3_cmd, "Stage 3: Save Crops (Optimized)")
        else:
            stage3_cmd = [
                sys.executable, "stage3_save_crops.py",
                "--metadata_file", args.metadata_file,
                "--dataset", args.dataset,
                "--split", args.split,
                "--output_dir", args.output_dir,
                "--start_idx", str(args.start_idx),
            ]
            if args.max_samples:
                stage3_cmd.extend(["--max_samples", str(args.max_samples)])
            if args.cpu_workers_stage3:
                stage3_cmd.extend(["--num_workers", str(args.cpu_workers_stage3)])

            success = run_command(stage3_cmd, "Stage 3: Save Crops")

        if not success:
            return 1
    else:
        print("\n⊘ Skipping Stage 3 (saving crops)")

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Preprocessed tensors: {args.tensor_dir}")
    print(f"Metadata file: {args.metadata_file}")
    print(f"Cropped images: {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
