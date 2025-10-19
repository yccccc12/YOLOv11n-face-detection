import cv2
import os
from ultralytics import YOLO
import time

# ====== Configuration ======
VIDEO_PATH = 'videos/vid2.mp4' # Path to input video
MODEL_PATH = 'models/best.pt'  # Path to the trained model
OUTPUT_PATH = 'outputs'  # Path to save output video
CONF_THRESHOLD = 0.5           # Confidence threshold
OUTPUT_SIZE = (960, 540)       # Output video resolution

# ====== Create output directory if it doesn't exist ======
os.makedirs(OUTPUT_PATH + '/track', exist_ok=True)

# ====== Initialize model and video capture ======
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter (MP4 format) to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
out = cv2.VideoWriter(OUTPUT_PATH + f'/track/{vid_name}_counting.mp4', fourcc, 30, OUTPUT_SIZE)


# ====== People counting setup ======
# Set line position to middle of frame
LINE_POSITION = width // 2

# Setup window
cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("People Counter", OUTPUT_SIZE[0], OUTPUT_SIZE[1])

# Counters
count_in = count_out = 0
track_history = {}
frame_idx = 0
start_time = time.time()

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
THICKNESS = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Perform tracking
    results = model.track(frame, conf=CONF_THRESHOLD, imgsz=640, persist=True, verbose=False)

    # Process results if there are tracked boxes
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2) # Calculate center x-coordinate

            # Tracking and counting
            if track_id not in track_history:
                track_history[track_id] = cx
            else:
                prev_x = track_history[track_id]

                # Check if person crossed the line
                if prev_x < LINE_POSITION <= cx:
                    count_in += 1 # cross from Left to Right
                elif prev_x > LINE_POSITION >= cx:
                    count_out += 1 # cross from Right to Left
                
                track_history[track_id] = cx

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

            # Draw label with confidence
            label = f"face {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
            
            # Positioning label above the box
            label_y = int(y1) - 10 if int(y1) > 30 else int(y2) + 25
            label_x = int(x1)
            
            # Draw red background box for label
            cv2.rectangle(frame,
                         (max(0, label_x - 8), max(0, label_y - label_h - 8)),
                         (min(width - 1, label_x + label_w + 8), min(height - 1, label_y + 6)),
                         (0, 0, 255), -1)
            
            # Put white text label on red background box
            cv2.putText(frame, label, (label_x + 4, label_y - baseline), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)

    # Draw counting line
    cv2.line(frame, (LINE_POSITION, 0), (LINE_POSITION, height), (255, 0, 0), 6)
    cv2.putText(frame, 'Line', (LINE_POSITION + 10, 30), FONT, 1.2, (255, 0, 0), 4)

    # Display counts statistics
    cv2.putText(frame, f'IN (Left to Right): {count_in}', (10, 30), FONT, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f'OUT (Right to Left): {count_out}', (10, 70), FONT, 1.2, (0, 0, 255), 2)

    resized_frame = cv2.resize(frame, OUTPUT_SIZE)
    out.write(resized_frame)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"IN (Left to Right): {count_in}, OUT (Right to Left): {count_out}")
print(f"Total: {count_in + count_out}")
print(f"FPS: {frame_idx / elapsed:.2f}")