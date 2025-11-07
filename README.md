# Bike Helmet Detection using YOLOv8

A computer vision project that detects whether bike riders are wearing helmets or not using YOLOv8 and OpenCV. The system can process both static images and real-time video streams.

## 🎯 Features

- **Image Detection**: Detect helmets in static images
- **Video Detection**: Real-time helmet detection in videos
- **Webcam Support**: Live detection using webcam feed
- **Custom Trained Model**: YOLOv8 model trained on bike helmet dataset
- **Visual Indicators**: Bounding boxes with confidence scores and labels

## 📋 Requirements

- Python 3.6 or later
- CUDA-compatible GPU (recommended for training)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ismailnossam01/bike-helmet-detection.git
cd bike-helmet-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install gitpython>=3.1.30
pip install matplotlib>=3.3
pip install numpy>=1.23.5
pip install opencv-python>=4.1.1
pip install pillow>=10.3.0
pip install psutil 
pip install PyYAML>=5.3.1
pip install requests>=2.32.0
pip install scipy>=1.4.1
pip install thop>=0.1.1
pip install torch>=1.8.0
pip install torchvision>=0.9.0
pip install tqdm>=4.64.0
pip install ultralytics>=8.2.34
pip install pandas>=1.1.4
pip install seaborn>=0.11.0
pip install setuptools>=65.5.1
pip install filterpy
pip install scikit-image
pip install lap
pip install cvzone
```

## 📁 Project Structure

```
BikeHelmetDetector/
├── Weights/
│   └── best.pt              # Trained YOLOv8 model
├── Media/
│   ├── riders_1.jpg         # Sample images
│   ├── riders_2.jpg
│   ├── traffic.mp4          # Sample video
│   └── ...
├── helmet_detector_image.py  # Image detection script
├── helmet_detector_video.py  # Video detection script
├── requirements.txt
└── README.md
```

## 🚀 Usage

### Image Detection

```python
python helmet_detector_image.py
```

Press `q` to close the detection window.

### Video Detection

```python
python helmet_detector_video.py
```

Press `q` to stop the video and close the window.

### Webcam Detection

Modify the video capture line in `helmet_detector_video.py`:
```python
# Replace this line
cap = cv2.VideoCapture("Media/traffic.mp4")

# With this
cap = cv2.VideoCapture(0)  # 0 for default camera
```

## 🎓 Training Your Own Model

### 1. Download Dataset
Download the bike helmet detection dataset from [Roboflow](https://roboflow.com/) and unzip it.

### 2. Setup Google Colab
- Open [Google Colab](https://colab.research.google.com/)
- Select Runtime → Change runtime type → T4 GPU
- Install Ultralytics: `!pip install ultralytics`

### 3. Upload Dataset to Google Drive
- Create folder structure: `MyDrive/Datasets/BikeHelmet/`
- Upload dataset contents to this folder
- Update `data.yaml` path to: `../drive/MyDrive/Datasets/BikeHelmet`

### 4. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Train the Model
```python
!yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/BikeHelmet/data.yaml epochs=100 imgsz=640
```

### 6. Download Trained Model
After training completes (1-2 hours), download `best.pt` from:
`runs/detect/train/weights/best.pt`

Place it in the `Weights/` folder of your project.

## 📊 Model Performance

- **Model**: YOLOv8l (Large)
- **Training Epochs**: 100
- **Image Size**: 640x640
- **Classes**: 
  - With Helmet
  - Without Helmet

## 🖼️ Sample Outputs

The system draws bounding boxes around detected helmets with:
- **Green boxes**: Detection boundaries
- **Labels**: "With Helmet" or "Without Helmet"
- **Confidence scores**: Detection accuracy (0.0 to 1.0)

## 🛠️ Code Explanation

### Key Components

1. **YOLO Model Loading**
```python
model = YOLO("Weights/best.pt")
```

2. **Object Detection**
```python
results = model(img, stream=True)
```

3. **Bounding Box Drawing**
```python
cvzone.cornerRect(img, (x1, y1, w, h))
cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, y1 - 10))
```

## ⚠️ Troubleshooting

**Issue**: Model not loading
- **Solution**: Ensure `best.pt` is in the `Weights/` folder

**Issue**: Video not opening
- **Solution**: Check video path in `Media/` folder

**Issue**: Low FPS during detection
- **Solution**: Use a GPU-enabled machine or reduce video resolution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset resources
- [OpenCV](https://opencv.org/) for image processing capabilities
- [PySeek](https://pyseek.com/) for the original tutorial

## 📚 Resources
- [Project Documentation]([https://docs.ultralytics.com/](https://pyseek.com/2024/08/bike-helmet-detection-project-in-python/))
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Original Tutorial](https://pyseek.com/computer-vision-projects/bike-helmet-detection/)

## 🔮 Future Enhancements

- [ ] Add support for multiple vehicle types
- [ ] Implement license plate detection for violators
- [ ] Create a web interface for easy access
- [ ] Add alert system for detected violations
- [ ] Improve model accuracy with more training data
- [ ] Deploy as a mobile application

---

⭐ If you found this project helpful, please give it a star!

