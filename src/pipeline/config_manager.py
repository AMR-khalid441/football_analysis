import os
import logging
import cv2
from utils.parameter_scaler import scale_for_resolution

logger = logging.getLogger("football_analysis")

def get_video_resolution(video_path):
    """
    Get video resolution safely.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple (width, height) or None if detection fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width, height)
    except Exception as e:
        logger.warning(f"Could not detect video resolution: {e}")
    return None

def setup_configuration():
    """
    Extract lines 22-60 from main.py
    Configuration constants, paths, and file validation
    """
    # -------------------------------
    # Configuration constants
    # -------------------------------
    MIN_PLAYERS_FOR_KMEANS = 6
    BALL_MAX_GAP_FRAMES = 15
    
    # Team assignment configuration
    TEAM_CONFIDENCE_THRESHOLD = 0.6
    TEAM_HYSTERESIS_FRAMES = 5
    TEAM_MIN_COLOR_SEPARATION = 50.0
    
    # -------------------------------
    # Base directory
    # -------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)  # Go up one level from pipeline folder

    # -------------------------------
    # Paths
    # -------------------------------
    input_path = os.path.join(script_dir, "input_videos", "CV_Task.mkv")
    output_path = os.path.join(script_dir, "output_videos", "output_video.avi")
    model_path = os.path.join(script_dir, "models", "best.pt")
    track_stub_path = os.path.join(script_dir, "stubs", "track_stubs.pkl")
    camera_stub_path = os.path.join(script_dir, "stubs", "camera_movement_stub.pkl")
    
    # -------------------------------
    # Resolution Detection
    # -------------------------------
    video_resolution = get_video_resolution(input_path)
    logger.info(f"Video resolution: {video_resolution}")

    # -------------------------------
    # Checks
    # -------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not os.path.exists(track_stub_path):
        logger.warning(f"Track stub not found: {track_stub_path} (will regenerate if needed)")
    if not os.path.exists(camera_stub_path):
        logger.warning(f"Camera movement stub not found: {camera_stub_path} (will regenerate if needed)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return {
        'MIN_PLAYERS_FOR_KMEANS': MIN_PLAYERS_FOR_KMEANS,
        'BALL_MAX_GAP_FRAMES': BALL_MAX_GAP_FRAMES,
        'TEAM_CONFIDENCE_THRESHOLD': TEAM_CONFIDENCE_THRESHOLD,
        'TEAM_HYSTERESIS_FRAMES': TEAM_HYSTERESIS_FRAMES,
        'TEAM_MIN_COLOR_SEPARATION': scale_for_resolution(TEAM_MIN_COLOR_SEPARATION, video_resolution),
        'input_path': input_path,
        'output_path': output_path,
        'model_path': model_path,
        'track_stub_path': track_stub_path,
        'camera_stub_path': camera_stub_path
    }
