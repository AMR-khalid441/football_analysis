# 📖 User Guide

> **Complete Setup and Usage Guide for Football Analysis System**

## 📋 Table of Contents

1. [Requirements & Installation](#requirements--installation)
2. [Model Download](#model-download)
3. [Sample Outputs](#sample-outputs)
4. [How to Run](#how-to-run)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## ⚙️ Requirements & Installation

### **System Requirements**
- Python 3.10 or above
- pip (latest version)
- Conda (recommended for environment management)

### **Installation Steps**

#### **1️⃣ Create Conda Environment**
```bash
conda create -n football_analysis python=3.11
conda activate football_analysis
```

#### **2️⃣ Install Dependencies**
```bash
# Install all requirements at once
pip install -r requirements.txt

# Or install individually
pip install ultralytics supervision opencv-python numpy pandas scikit-learn
```

#### **3️⃣ Verify Installation**
```bash
python -c "import ultralytics; print('YOLOv5 installed successfully')"
```

---

## 📦 Model Download

### **Download Trained YOLOv5 Model**

The system requires a pre-trained YOLOv5 model for player, referee, and ball detection.

👉 **[Download YOLOv5 Model (best.pt)](https://drive.google.com/file/d/1XVBKxLP5DHxaqX896YocF_476G0pH8FE/view?usp=drive_link)**

### **Model Placement**
After downloading, place the model file in the correct location:

```
football_analysis/
├── models/
│   └── best.pt          # ← Place your model here
├── input_videos/
├── output_videos/
└── src/
```

### **Model Training & Development**
For those interested in training their own model or understanding the development process:

🔗 **[Football YOLOv5 Training Notebook](https://drive.google.com/drive/folders/1zk8Dbs9FHfxVOY8N28milCaVb83QV31t?usp=drive_link)**

#### **What's Included:**
- **📊 Dataset Preparation**: Complete guide to data collection and labeling techniques
- **⚙️ Model Configuration**: YOLOv5 setup, hyperparameter tuning, and fine-tuning process  
- **🚀 Training Pipeline**: Step-by-step training workflow with best practices
- **📈 Evaluation Metrics**: Comprehensive performance analysis and validation methods
- **🔧 Custom Model Creation**: Build your own specialized detection model for different scenarios
- **📝 Code Examples**: Ready-to-use code snippets and implementations
- **🎯 Optimization Tips**: Performance tuning and model improvement strategies

---

## 🎥 Sample Outputs

### **Output Videos**

The system generates annotated videos showing:
- Player detection and tracking
- Team color assignment
- Ball tracking and control
- Speed and distance analysis
- Camera motion compensation

#### **Sample Videos**
🎥 **[Sample Output Video 1](https://drive.google.com/file/d/1k2Qxd9zia1oz50TwX6sha8my_zZY7o-R/view?usp=sharing)** - Full match view with player and referee detection (Output before enhancements)

🎥 **[Sample Output Video 2](https://drive.google.com/file/d/1LrR24f6LFKTV6bF0Ea3Ajp_T-xs0x9R4/view?usp=sharing)** - Local demo showing complete pipeline (Output before enhancements)

🎥 **[Enhanced Output Video](https://drive.google.com/file/d/1fcuZMbf07YP7Rkro5sgi_cSQtp6wlX8D/view?usp=drive_link)** - Complete analysis with all enhancements and optimizations applied

### **Output Features**
- **Player Detection**: Bounding boxes around all players
- **Team Assignment**: Color-coded team identification
- **Ball Tracking**: Ball position and control assignment
- **Speed Analysis**: Player speed measurements
- **Referee Detection**: Referee identification and tracking

---

## 🏁 How to Run

### **Step 1: Prepare Input Video**
```bash
# Place your video in the input directory
cp your_video.mp4 input_videos/
```

### **Step 2: Verify Model**
```bash
# Ensure model is in correct location
ls models/best.pt
```

### **Step 3: Run Analysis**
```bash
# Activate environment
conda activate football_analysis

# Run the analysis
python main.py
```

### **Step 4: Check Output**
```bash
# Output will be saved to output_videos/
ls output_videos/
```

---

## ⚙️ Configuration

### **Basic Configuration**
The system uses configuration parameters that can be modified in `src/pipeline/config_manager.py`:

```python
# Key parameters
MIN_PLAYERS_FOR_KMEANS = 6
BALL_MAX_GAP_FRAMES = 15
TEAM_CONFIDENCE_THRESHOLD = 0.6
TEAM_HYSTERESIS_FRAMES = 5
```

### **Resolution-Based Scaling**
Parameters automatically scale based on video resolution:
- 720p: Parameters scaled down
- 1080p: Standard parameters
- 4K: Parameters scaled up

### **Custom Configuration**
```python
# Modify parameters in config_manager.py
config = {
    'TEAM_CONFIDENCE_THRESHOLD': 0.8,  # Higher confidence
    'SPEED_SMOOTHING_ALPHA': 0.5,      # More responsive smoothing
}
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Model Not Found**
```
Error: Model file not found
Solution: Ensure best.pt is in models/ directory
```

#### **2. Memory Issues**
```
Error: Out of memory
Solution: Reduce batch size in detection_processor.py
```

#### **3. Video Format Issues**
```
Error: Cannot open video
Solution: Convert to MP4 format using ffmpeg
```

#### **4. Dependencies Missing**
```
Error: Module not found
Solution: pip install -r requirements.txt
```

### **Performance Tips**
- Use SSD storage for faster video I/O
- Ensure sufficient RAM (8GB+ recommended)
- Close other applications during processing
- Use GPU acceleration if available

---

## 📈 Next Steps

### **Planned Features**
1. **Jersey Number Recognition**
   - OCR/CNN module for number detection
   - Player identification system
   - Jersey number tracking

2. **Advanced Analytics**
   - Heat maps for player movement
   - Pass analysis and statistics
   - Formation detection

3. **Real-time Processing**
   - Live video stream analysis
   - Real-time statistics
   - Web interface

### **Customization**
- Modify detection confidence thresholds
- Add custom team colors
- Implement additional statistics
- Create custom visualizations

---

*For technical details, see [API Documentation](API_DOCS.md) and [Architecture Guide](ARCHITECTURE.md).*
