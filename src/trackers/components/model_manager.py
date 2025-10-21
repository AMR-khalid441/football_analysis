from ultralytics import YOLO
import supervision as sv

class ModelManager:
    def __init__(self, model_path):
        """
        Extract lines 13-25 from tracker.py
        Model loading with error handling and fallback
        """
        try:
            self.model = YOLO(model_path) 
            self.tracker = sv.ByteTrack()
            print(f"✅ Successfully loaded custom model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            print("🔄 Trying to use a smaller model as fallback...")
            # Fallback to a smaller model
            self.model = YOLO('yolov5su.pt')  # Much smaller than yolov5x
            self.tracker = sv.ByteTrack()
            print("✅ Successfully loaded fallback model: yolov5su.pt")
            print("⚠️  Note: Using COCO classes instead of custom football classes")
