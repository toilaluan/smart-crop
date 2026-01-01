"""
Smart Image Cropper using DINOv2 for feature-based similarity matching.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import Tuple, List, Optional
import numpy as np


class SmartImageCropper:
    """
    Crops non-square images to square by finding the most semantically important region
    using DINOv2 feature similarity.
    """

    def __init__(
        self, model_name: str = "facebook/dinov2-vits16", device: Optional[str] = None
    ):
        """
        Initialize the smart cropper with DINOv2 model.

        Args:
            model_name: HuggingFace model identifier for DINOv2
            device: Device to run model on. If None, uses auto device mapping
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading DINOv2 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, device_map="auto" if device is None else None
        )

        if device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

    def _compute_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Compute DINOv2 pooled feature for an image.

        Args:
            image: PIL Image

        Returns:
            Pooled feature tensor
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.pooler_output

    def _compute_features_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Compute DINOv2 pooled features for a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Batched pooled feature tensor of shape (batch_size, feature_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.pooler_output

    def _cosine_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            feat1: First feature tensor
            feat2: Second feature tensor

        Returns:
            Cosine similarity score
        """
        feat1_norm = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2_norm = feat2 / feat2.norm(dim=-1, keepdim=True)

        similarity = (feat1_norm * feat2_norm).sum(dim=-1)
        return similarity.item()

    def _cosine_similarity_batch(
        self, feat_original: torch.Tensor, feat_crops: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between original feature and batch of crop features.

        Args:
            feat_original: Original image feature tensor of shape (1, feature_dim)
            feat_crops: Crop features tensor of shape (num_crops, feature_dim)

        Returns:
            Similarity scores tensor of shape (num_crops,)
        """
        feat_original_norm = feat_original / feat_original.norm(dim=-1, keepdim=True)
        feat_crops_norm = feat_crops / feat_crops.norm(dim=-1, keepdim=True)

        # Compute similarity: (num_crops, feature_dim) @ (feature_dim, 1) -> (num_crops,)
        similarities = (feat_crops_norm * feat_original_norm).sum(dim=-1)
        return similarities

    def _generate_crops(
        self, image: Image.Image, crop_size: int, num_crops: int
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Generate N evenly-spaced crops along the longer dimension.

        Args:
            image: Original PIL Image
            crop_size: Size of square crop (min of width/height)
            num_crops: Number of crops to generate

        Returns:
            List of tuples (cropped_image, (left, top, right, bottom))
        """
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

    def crop(
        self,
        image: Image.Image,
        num_crops: int = 16,
        return_all_scores: bool = False,
        resize_to: Optional[int] = None,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
        """
        Find and return the best square crop of the image.

        Args:
            image: Input PIL Image (non-square)
            num_crops: Number of candidate crops to evaluate
            return_all_scores: If True, also return all crop scores

        Returns:
            Tuple of (best_cropped_image, crop_box, similarity_score)
            If return_all_scores is True, returns additional list of all scores
        """
        width, height = image.size

        # resize shorter side to resize_to if specified
        if resize_to is not None:
            if width >= height:
                new_height = resize_to
                new_width = int(width * (resize_to / height)) // 16 * 16
            else:
                new_width = resize_to
                new_height = int(height * (resize_to / width)) // 16 * 16
            image = image.resize((new_width, new_height), Image.LANCZOS)
            width, height = image.size
            print(f"Resized image to: {width}x{height}")

        if width == height:
            print("Image is already square, returning original")
            if return_all_scores:
                return image, (0, 0, width, height), 1.0, [(1.0, (0, 0, width, height))]
            return image, (0, 0, width, height), 1.0

        crop_size = min(width, height)
        print(f"Original size: {width}x{height}, Crop size: {crop_size}x{crop_size}")

        # Generate candidate crops
        print(f"Generating {num_crops} candidate crops...")
        crops = self._generate_crops(image, crop_size, num_crops)

        # Prepare batch: original image + all crops
        print("Computing features for original and all crops in batch...")
        images_batch = [image] + [cropped_img for cropped_img, _ in crops]

        # Compute features for all images in one batch
        all_features = self._compute_features_batch(images_batch)

        # Split: first feature is original, rest are crops
        original_feature = all_features[0:1]  # Keep shape (1, feature_dim)
        crop_features = all_features[1:]  # Shape (num_crops, feature_dim)

        # Compute similarities in batch
        print("Computing similarities...")
        similarities = self._cosine_similarity_batch(original_feature, crop_features)

        # Pair similarities with boxes and crops
        scores = []
        for i, ((cropped_img, box), similarity) in enumerate(zip(crops, similarities)):
            similarity_value = similarity.item()
            scores.append((similarity_value, box, cropped_img))
            print(
                f"  Crop {i + 1}/{num_crops} - Box: {box}, Similarity: {similarity_value:.4f}"
            )

        # Find best crop
        best_similarity, best_box, best_crop = max(scores, key=lambda x: x[0])
        print(f"\nBest crop: {best_box} with similarity {best_similarity:.4f}")

        if return_all_scores:
            all_scores = [(s, b) for s, b, _ in scores]
            return best_crop, best_box, best_similarity, all_scores

        return best_crop, best_box, best_similarity

    def crop_from_path(
        self,
        image_path: str,
        num_crops: int = 5,
        save_path: Optional[str] = None,
        resize_to: Optional[int] = None,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
        """
        Load image from path and crop it.

        Args:
            image_path: Path to input image
            num_crops: Number of candidate crops to evaluate
            save_path: Optional path to save the cropped image

        Returns:
            Tuple of (best_cropped_image, crop_box, similarity_score)
        """
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # extract size from image
        H, W = image.size
        if resize_to is not None and max(H, W) > resize_to:
            if H >= W:
                new_H = resize_to
                new_W = int(W * (resize_to / H))
            else:
                new_W = resize_to
                new_H = int(H * (resize_to / W))
            image = image.resize((new_W, new_H), Image.LANCZOS)

        result = self.crop(image, num_crops=num_crops)
        best_crop, best_box, best_similarity = result

        if save_path:
            print(f"Saving cropped image to: {save_path}")
            best_crop.save(save_path)

        return result


if __name__ == "__main__":
    # Example usage
    from transformers.image_utils import load_image

    # Load a test image
    url = "https://images.pexels.com/photos/3243/pen-calendar-to-do-checklist.jpg?cs=srgb&dl=pexels-breakingpic-3243.jpg&fm=jpg"
    image = load_image(url)

    # Create cropper
    cropper = SmartImageCropper(
        "facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda"
    )

    # Crop the image
    cropped_image, crop_box, similarity = cropper.crop(
        image, num_crops=16, resize_to=1024
    )

    print(f"\nFinal result:")
    print(f"  Crop box: {crop_box}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Cropped size: {cropped_image.size}")

    # Save result
    cropped_image.save("cropped_result.jpg")
    print("Saved to cropped_result.jpg")
