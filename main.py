import os
from datasets import load_dataset
import torch
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

# Constants
NUM_CROPS = 15
PRETRAINED_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SAVE_DIR = "best_crops"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load processor and model
processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME, device_map="auto")

# Load dataset
ds = load_dataset("ujin-song/pexels-image-60k", split="train", num_proc=16)
ds = ds.select_columns(["image"])


def resize_image(image, size=1024):
    """Resize image while preserving aspect ratio, ensuring dimensions are multiples of 16."""
    width, height = image.size
    if width > height:
        new_width = size
        new_height = (size * height // width) // 16 * 16
    else:
        new_height = size
        new_width = (size * width // height) // 16 * 16
    return image.resize((new_width, new_height))


def generate_crops(image, num_crops=NUM_CROPS):
    """Generate sliding square crops and one central square resize."""
    w, h = image.size
    crops = []
    if w > h:
        stride = (w - h) / (num_crops - 1)
        for i in range(num_crops):
            left = int(i * stride)
            crop = image.crop((left, 0, left + h, h))
            crops.append(crop)
    else:
        stride = (h - w) / (num_crops - 1)
        for i in range(num_crops):
            top = int(i * stride)
            crop = image.crop((0, top, w, top + w))
            crops.append(crop)
    # Append the square resize
    min_side = min(w, h)
    crops.append(image.resize((min_side, min_side)))
    return crops


# Preprocess dataset: Add resized images
ds = ds.map(
    lambda x: {
        "image_1024": resize_image(x["image"], size=1024),
        "image_224": resize_image(x["image"], size=224),
    },
    num_proc=16,
)

# Add crops from 224-resized image
ds = ds.map(
    lambda x: {"crops_224": generate_crops(x["image_224"])},
    num_proc=16,
    remove_columns=["image_224"],
)

# Process crops to tensor
ds = ds.map(
    lambda x: {
        "pixel_values": processor(
            images=x["crops_224"], return_tensors="pt"
        ).pixel_values
    },
    num_proc=16,
    remove_columns=["crops_224"],
)

# Set format to torch for compatible fields
ds = ds.with_format("torch")

# DataLoader for inference
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

# Inference loop to find best crop indices and save best crops
for idx, batch in enumerate(dl):
    image_1024 = batch["image_1024"][0]  # PIL Image
    pixel_values = batch["pixel_values"][0].to(
        model.device
    )  # (num_crops + 1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        features = outputs.pooler_output  # (num_crops + 1, D)
        original_feature = features[-1].unsqueeze(0)  # (1, D)
        crop_features = features[:-1]  # (num_crops, D)
        similarities = torch.nn.functional.cosine_similarity(
            original_feature, crop_features
        )  # (num_crops,)
        best_idx = torch.argmax(similarities).item()

    # Generate crops on 1024-resized image and select the best one
    crops_1024 = generate_crops(image_1024)
    best_crop = crops_1024[best_idx]

    # Save the best crop
    save_path = os.path.join(SAVE_DIR, f"image_{idx}.jpg")
    best_crop.save(save_path)

    print(f"Saved best crop for image {idx} to {save_path}")
