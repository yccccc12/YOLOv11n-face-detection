import cv2
import os
from ultralytics import solutions

VIDEO_PATH = "videos/vid2.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_PATH = "outputs/track"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error opening video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Vertical line in the middle
middle_x = w // 2
region_points = [(middle_x, 0), (middle_x, h)]

# Video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
save_path = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
video_writer = cv2.VideoWriter(OUTPUT_PATH + f'/{save_path}_counting.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

# Initialize the object counter
counter = solutions.ObjectCounter(model=MODEL_PATH, region=region_points, show=False, conf=0.5)

# Scale for display
display_scale = 0.7

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)
    frame = results.plot_im

    # Get current counts
    in_count = counter.in_count
    out_count = counter.out_count

    # Resize frame for display
    display_frame = cv2.resize(frame, (int(w * display_scale), int(h * display_scale)))

    # Show resized frame
    cv2.imshow("Object Counting", display_frame)

    # Write original size to output video
    video_writer.write(frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Clean up
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Print final count
print(f"Final In Count: {counter.in_count}")
print(f"Final Out Count: {counter.out_count}")
