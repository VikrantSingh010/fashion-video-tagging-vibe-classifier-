import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ---------------------------- Setup ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

ckpt = 'yainage90/fashion-object-detection'
image_processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)

# ---------------------------- Constants ----------------------------
FRAME_ROOT = 'frames'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------- Helper ----------------------------
def process_image(img_path):
    image = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
        results = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        bbox = box.tolist()
        bbox = [round(bbox[0], 2), round(bbox[1], 2), round(bbox[2] - bbox[0], 2), round(bbox[3] - bbox[1], 2)]  # x, y, w, h
        detections.append({
            "class_name": label_name,
            "confidence": round(score.item(), 3),
            "bbox": bbox
        })
    return detections

# ---------------------------- Main Loop ----------------------------
for video_folder in tqdm(sorted(os.listdir(FRAME_ROOT))):
    folder_path = os.path.join(FRAME_ROOT, video_folder)
    if not os.path.isdir(folder_path):
        continue

    video_results = {
        "video_id": video_folder,
        "frames": []
    }

    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame_number = int(''.join(filter(str.isdigit, frame_file)))
        detections = process_image(frame_path)

        for det in detections:
            det["frame_number"] = frame_number

        if detections:
            video_results["frames"].append({
                "frame_file": frame_file,
                "detections": detections
            })

    # Save result
    out_path = os.path.join(OUTPUT_DIR, f"{video_folder}.json")
    with open(out_path, 'w') as f:
        json.dump(video_results, f, indent=2)

    print(f"Saved detections to {out_path}")
