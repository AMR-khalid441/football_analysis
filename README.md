# Football Analysis System

This project is a football match analysis system built using YOLOv8 and computer vision techniques.  
It detects players, referees, and the ball, identifies teams based on colors, and tracks movements across the field.

---

## ⚙️ Requirements

To run this application, make sure you have the following installed:

- Python 3.10 or above
- pip (latest version)
- The following libraries:
  ```bash
  pip install ultralytics supervision 📦 Requirements & Installation

Before starting, make sure you have Conda installed (via Anaconda
 or Miniconda
).

1️⃣ Create a Conda environment

```bash
conda create -n football_analysis python=3.11
```

activate it ..
```bash
conda activate football_analysis
```
2️⃣ Install the required libraries


to install everything at once (recommended for running the full project):




```bash
Copy code
pip install -r requirements.txt
```
📓 Model Notebook (Training / Fine-tuning)
You can view the notebook used for model training and fine-tuning here:

🔗 Football YOLOv8 Model Notebook (Conda)

This notebook includes:

Dataset preparation

YOLOv5 model training

Evaluation and inference examples

🎥 Output Videos
Below are three output samples demonstrating the system:

Video	Description
Output 1	Full match view — detects players and referees


(If GitHub doesn’t show video previews, you can upload short GIFs or screenshots instead.)


🔧 Components & Flow Diagram
Components Overview
YOLOv5 Detector — Detects players, referees, and the ball

Tracker — Keeps consistent IDs across frames

Team Assigner — Differentiates between teams using dominant color clustering

Annotator — Draws detections and adds team and referee labels

Statistics Module — Calculates ball control and distance metrics

Flow Diagram:

📊



🏁 How to Run
Add your input video to the input_videos/ folder

Make sure your YOLO model is saved as models/best.pt

Run:

bash
Copy code
python main.py
Output will be saved to output_videos/ with all detections and tracking results

📈 Notes
The current version detects players, teams, and referees, but the ball label still needs fine-tuning.

The system runs frame-by-frame and saves a processed video automatically.

You can modify detection confidence or classes directly in main.py if needed.

👨‍💻 Author
Amr Khalid

AI Engineer — Computer Vision

📍 Cairo, Egypt

🔗 LinkedIn

🔗 GitHub








