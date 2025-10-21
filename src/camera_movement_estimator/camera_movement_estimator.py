from .components.optical_flow_config import OpticalFlowConfig
from .components.position_adjuster import PositionAdjuster
from .components.motion_detector import MotionDetector
from .components.motion_visualizer import MotionVisualizer

class CameraMovementEstimator():
    def __init__(self, frame=None):
        """
        Simplified CameraMovementEstimator class that uses modular components
        Maintains same interface as original camera_movement_estimator.py
        """
        # Initialize all components
        self.config = OpticalFlowConfig(frame)
        self.position_adjuster = PositionAdjuster()
        self.motion_detector = MotionDetector(self.config)
        self.motion_visualizer = MotionVisualizer()
        
        # Store first_frame reference for backward compatibility
        self.first_frame = frame

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Delegate to position_adjuster
        """
        return self.position_adjuster.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Delegate to motion_detector
        """
        return self.motion_detector.detect_camera_movement(frames, read_from_stub, stub_path)
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Delegate to motion_visualizer
        """
        return self.motion_visualizer.draw_camera_movement(frames, camera_movement_per_frame)