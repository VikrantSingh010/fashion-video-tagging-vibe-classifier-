import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm


CROPPED_DIR = 'cropped'

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()

clip_embeddings={}

for video_id in os.listdir(CROPPED_DIR):
    video_path=os.path.join(CROPPED_DIR,video_id)

    if not os.path.isdir(video_path):
        continue
    for img_name in tqdm(os.listdir(video_path),desc=f"Embedding {video_id}"):
        img_path=os.path.join(video_path,img_name)

        image=Image.open(img_path).convert("RGB")

        inputs=clip_processor(images=image, return_tensors="pt",padding=True)

        with torch.no_grad():
            image_features=clip_model.get_image_features(**inputs)
        embedding=image_features[0].cpu().numpy()

        clip_embeddings[f"{video_id}/{img_name}"]=embedding.tolist()



import json
with open("clip_embeddings.json","w") as f:
    json.dump(clip_embeddings,f)

print("All Embeddings saved to Clip_Embeddings")

