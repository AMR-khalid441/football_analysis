from .components.model_manager import ModelManager
from .components.detection_processor import DetectionProcessor
from .components.tracking_engine import TrackingEngine
from .components.visualizer import Visualizer
from .components.track_utils import add_position_to_tracks, interpolate_ball_positions

class Tracker:
    def __init__(self, model_path):
        """
        Simplified Tracker class that uses modular components
        Maintains same interface as original tracker.py
        """
        # Initialize all components
        self.model_manager = ModelManager(model_path)
        self.detection_processor = DetectionProcessor(self.model_manager.model)
        self.tracking_engine = TrackingEngine(self.model_manager.model, self.model_manager.tracker)
        self.visualizer = Visualizer()
        
        # Expose model and tracker for backward compatibility
        self.model = self.model_manager.model
        self.tracker = self.model_manager.tracker

    def add_position_to_tracks(self, tracks):
        """
        Delegate to track_utils function
        """
        return add_position_to_tracks(tracks)

    def interpolate_ball_positions(self, ball_positions):
        """
        Delegate to track_utils function
        """
        return interpolate_ball_positions(ball_positions)

    def detect_frames(self, frames):
        """
        Delegate to detection_processor
        """
        return self.detection_processor.detect_frames(frames)

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Delegate to tracking_engine
        """
        return self.tracking_engine.get_object_tracks(frames, read_from_stub, stub_path)

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Delegate to visualizer
        """
        return self.visualizer.draw_ellipse(frame, bbox, color, track_id)

    def draw_triangle(self, frame, bbox, color):
        """
        Delegate to visualizer
        """
        return self.visualizer.draw_triangle(frame, bbox, color)

    def draw_annotations_single_frame(self, frame, frame_num, tracks, team_ball_control):
        """
        Delegate to visualizer
        """
        return self.visualizer.draw_annotations_single_frame(frame, frame_num, tracks, team_ball_control)

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Delegate to visualizer
        """
        return self.visualizer.draw_annotations(video_frames, tracks, team_ball_control)