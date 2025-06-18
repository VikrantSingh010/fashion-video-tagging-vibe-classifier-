import os
import json
from PIL import Image

# Directories
FRAMES_DIR = 'frames'
OUTPUTS_DIR = 'outputs'
CROPPED_DIR = 'cropped'
os.makedirs(CROPPED_DIR, exist_ok=True)

for output_file in os.listdir(OUTPUTS_DIR):
    if not output_file.endswith('.json'):
        continue

    video_id = output_file.replace('.json', '')
    video_json_path = os.path.join(OUTPUTS_DIR, output_file)
    frame_dir = os.path.join(FRAMES_DIR, video_id)

    with open(video_json_path, 'r') as f:
        data = json.load(f)

    video_crop_dir = os.path.join(CROPPED_DIR, video_id)
    os.makedirs(video_crop_dir, exist_ok=True)

    for frame_data in data['frames']:
        frame_file = frame_data['frame_file']
        frame_path = os.path.join(frame_dir, frame_file)

        if not os.path.exists(frame_path):
            print(f"[WARN] Frame not found: {frame_path}")
            continue

        image = Image.open(frame_path).convert('RGB')

        for i, det in enumerate(frame_data['detections']):
            x, y, w, h = det['bbox']
            crop = image.crop((x, y, x + w, y + h))

            crop_filename = f"{os.path.splitext(frame_file)[0]}_det{i}_{det['class_name']}.jpg"
            crop_path = os.path.join(video_crop_dir, crop_filename)
            crop.save(crop_path)

print("✅ All detected objects cropped and saved in 'cropped/'")
