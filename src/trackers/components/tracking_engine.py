import pickle
import os
import supervision as sv
from .detection_processor import DetectionProcessor

class TrackingEngine:
    def __init__(self, model, tracker):
        """
        Extract lines 71-165 from tracker.py
        Object tracking and data structure management
        """
        self.model = model
        self.tracker = tracker
        self.detection_processor = DetectionProcessor(model)
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Extract lines 71-165 from tracker.py
        Main tracking logic with stub management
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detection_processor.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            # Handle case where detection failed due to memory error
            if detection is None:
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                continue
                
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            # Debug: Show available classes (only on first frame)
            if frame_num == 0:
                print(f"🔍 Available classes in model: {list(cls_names.keys())}")

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Handle different model class names gracefully
            # For custom models: ['ball', 'goalkeeper', 'player', 'referee']
            # For COCO models: ['person', 'sports ball', ...]
            player_class_id = None
            referee_class_id = None
            ball_class_id = None
            
            # Try to find appropriate class IDs
            if 'player' in cls_names_inv:
                player_class_id = cls_names_inv['player']
            elif 'goalkeeper' in cls_names_inv:
                player_class_id = cls_names_inv['goalkeeper']
            elif 'person' in cls_names_inv:  # COCO fallback
                player_class_id = cls_names_inv['person']
                
            if 'referee' in cls_names_inv:
                referee_class_id = cls_names_inv['referee']
            elif 'person' in cls_names_inv:  # COCO fallback - treat as referee
                referee_class_id = cls_names_inv['person']
                
            if 'ball' in cls_names_inv:
                ball_class_id = cls_names_inv['ball']
            elif 'sports ball' in cls_names_inv:  # COCO fallback
                ball_class_id = cls_names_inv['sports ball']

            # Convert GoalKeeper to player object (if using custom model)
            if 'goalkeeper' in cls_names_inv and 'player' in cls_names_inv:
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Use found class IDs or skip if not available
                if player_class_id is not None and cls_id == player_class_id:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if referee_class_id is not None and cls_id == referee_class_id:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if ball_class_id is not None and cls_id == ball_class_id:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
