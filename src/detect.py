from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from moviepy import VideoFileClip
import os

# ===== Configuration =====
MODEL_PATH = 'models/best.pt'
INPUT_PATH = 'images/img1.jpg'
OUTPUT_PATH = 'outputs'

# ===== Initialize and run model =====
model = YOLO(MODEL_PATH)
results = model(INPUT_PATH, save=True, project=OUTPUT_PATH)

# Find output file
output_dir = sorted(Path(OUTPUT_PATH).glob('predict*'))[-1]
output_file = list(output_dir.glob('*.*'))[0]

# Determine if input is image or video
is_image = Path(INPUT_PATH).suffix.lower() in {'.jpg', '.png', '.jpeg'}

# ===== Display results =====
if is_image:
    # Load and display image
    img = mpimg.imread(str(output_file))

    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.title('Model Prediction')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    # Convert AVI to MP4
    mp4_path = output_file.with_suffix('.mp4')
    print(f"Converting predicted AVI to MP4...")

    clip = VideoFileClip(str(output_file))
    clip.write_videofile(str(mp4_path), codec='libx264', audio_codec='aac', preset='medium', fps=clip.fps)
    clip.close()

    # Delete the original AVI file
    os.remove(output_file)

    # Update output file path with MP4 path
    print(f"Conversion done: {mp4_path}")
    output_file = mp4_path
    
    # Load and display video
    cap = cv2.VideoCapture(str(output_file))

    # Get video dimensions and set window size
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1280 / w, 720 / h)
    
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', int(w * scale), int(h * scale))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()