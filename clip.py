from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # Load CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # Load processor

img = Image.open("frames/2025-05-31_14-01-37_UTC/frame_0003.jpg").convert("RGB")  # Open and convert image
inputs = processor(images=img, return_tensors="pt")  # Preprocess image

with torch.no_grad():
    image_features = model.get_image_features(**inputs)  # Get image embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize embedding

print(image_features)  # Print the normalized embedding vector
