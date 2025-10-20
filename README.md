# YOLOv11n Face Detection & People Counting

A real-time face detection and people counting system using YOLOv11n, fine-tuned on a custom dataset using Roboflow platform. This project implements transfer learning, object tracking, and a counting zone to monitor people crossing a virtual line.

## Project Purpose

This project demonstrates:
1. **Custom Dataset Creation**: Generate and augment a face detection dataset using Roboflow
2. **Transfer Learning**: Fine-tune YOLOv11n model for face detection
3. **People Counting**: Implement a counting zone to track people crossing a virtual line
4. **Performance Evaluation**: Measure model performance using mAP and FPS metrics
5. **Robustness Testing**: Validate the solution on test video

## Project Structure

```
YOLOv11n-face-detection/
├── images/                          # Test images
├── models/
│   └── best.pt                      # Fine-tuned model weights
├── outputs/                         # Detection results
|   ├── predict/                     # Image or video predictions
|   └── track/                       # Tracking videos with counting zone
├── videos/                          # Test videos
├── counter.py                       # People counting with tracking
├── detect.py                        # Basic detection script
├── requirements.txt                 # Python dependencies
└── YOLOv11_Face_FineTuning.ipynb    # Training notebook
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yccccc12/YOLOv11n-face-detection.git
cd YOLOv11n-face-detection
```

### 2. Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Model

After running the training notebook, the fine-tuned model is downloaded (`best.pt`).
Remember to place the fine-tuned model weights (`best.pt`) in the `models/` directory.

## Dataset Preparation

The dataset is prepared through the Roboflow platform since it provide function like data augmentation, image collection from other user, easy to upload own data and so forth.

### Using Roboflow for Custom Dataset

1. **Collect Images**: 63 images are collected from different angles, lighting, backgrounds
2. **Annotate Faces**: All faces occured in images are labelled with bounding boxes
3. **Preprocessing**
   - Auto-Orient: Applied
   - Resize: Stretch to 640x640
   - Auto-Adjust Contrast: Using Contrast Stretching
4. **Augmentation Applied**:
   - Outputs per training example: 3
   - Flip: Horizontal
   - 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
   - Crop: 0% Minimum Zoom, 6% Maximum Zoom
   - Rotation: Between -10° and +10°
   - Shear: ±10° Horizontal, ±10° Vertical
   - Saturation: Between -10% and +10%
   - Brightness: Between -15% and +15%
   - Exposure: Between -10% and +10%
   - Blur: Up to 1.5px
   - Noise: Up to 0.1% of pixels
   ** After augementation, there are total 133 iamges
5. **Dataset Split**: Train set (79%), Valid set (10%) and test set (11%)

### Augmentation Benefits

- Increases dataset size by 3-5x
- Improves model generalization
- Reduces overfitting
- Enhances robustness to various conditions

## Model Training

### Fine-Tuning YOLOv11n

This project uses a pre-trained YOLOv11n face detection model from Hugging Face as the starting point for transfer learning.

**Base Model**: [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)

**Regularization Applied:**
- Start with pretrained weights (transfer learning)
- Use early stopping to prevent overfitting
- Monitor validation mAP during training
- Adjust learning rate if needed

## Usage

### 1. Basic Face Detection

Run detection on images or videos using [`detect.py`](detect.py):

```bash
python detect.py
```

**Configuration** (edit in `detect.py`):
```python
MODEL_PATH = 'models/best.pt'
INPUT_PATH = 'videos/vid1.mp4'  # or 'images/img1.jpg'
OUTPUT_PATH = 'outputs'
```

### 2. People Counting with Tracking

Run people counting using [`counter.py`](counter.py):

```bash
python counter.py
```

**Configuration** (edit in `counter.py`):
```python
MODEL_PATH = 'models/best.pt'
INPUT_PATH = 'videos/vid1.mp4' # editable
OUTPUT_PATH = 'outputs'
```

**Features:**
- Virtual counting line at frame center
- IN counter (left to right crossing)
- OUT counter (right to left crossing)
- Track ID assignment

**Controls:**
For video prediction:
- Press `q` to stop video playback
- Results automatically saved to output directory

## Performance Metrics

Processing.....

### Mean Average Precision (mAP)

Evaluated on validation set:
- **mAP@0.5**: Measures detection accuracy at 50% IoU threshold
- **mAP@0.5:0.95**: Average mAP across IoU thresholds from 0.5 to 0.95

### Frames Per Second (FPS)

Real-time processing speed:
- Measured during video inference
- Displayed in console output after processing
- Format: `FPS: XX.XX`

**Example Output:**
```
IN (Left to Right): 15, OUT (Right to Left): 12
Total: 27
FPS: 28.45
```

## Detection Results

The detection results are saved under the `outputs/` folder. \
-> `outputs/predict` saved the result for normal face detection like images or videos\
-> `outputs/track` saved the video of people counting prediction

You are feel free to try on your own images or videos but make sure upload them to the corresponding subfolder and remember to modify the input path name under the `detect.py` or `counter.py` scripts.

## License

This project is for educational purposes.

## Acknowledgments

- [Huggingface YOLOv11](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)
- [Roboflow](https://roboflow.com/) for dataset management
- OpenCV community
