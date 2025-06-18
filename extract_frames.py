import cv2
import os

def extract_frames_from_all_videos(video_dir='videos', output_dir='frames', every_n_frames=10):
    # Make sure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            video_id = os.path.splitext(video_file)[0]  # e.g., reel_001
            output_path = os.path.join(output_dir, video_id)
            os.makedirs(output_path, exist_ok=True)
            print(f"[INFO] Processing {video_file}")

            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % every_n_frames == 0:
                    filename = f"frame_{saved:04d}.jpg"
                    full_path = os.path.join(output_path, filename)
                    cv2.imwrite(full_path, frame)
                    saved += 1
                frame_count += 1
            cap.release()
            print(f"[✓] Saved {saved} frames from {video_file} into {output_path}")

# Run the function
extract_frames_from_all_videos(every_n_frames=10)
