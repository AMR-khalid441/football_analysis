from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def initialize_modules(config):
    """
    Extract lines 62-76 from main.py
    Initialize all AI modules with configuration
    """
    # -------------------------------
    # Initialize modules
    # -------------------------------
    tracker = Tracker(config['model_path'])
    camera_estimator = CameraMovementEstimator(None)  # frame will be set later
    team_assigner = TeamAssigner()
    
    # Configure team assigner with thresholds
    team_assigner.confidence_threshold = config['TEAM_CONFIDENCE_THRESHOLD']
    team_assigner.hysteresis_frames = config['TEAM_HYSTERESIS_FRAMES']
    team_assigner.min_color_separation = config['TEAM_MIN_COLOR_SEPARATION']
    
    player_assigner = PlayerBallAssigner()
    view_transformer = ViewTransformer()
    speed_distance_estimator = SpeedAndDistance_Estimator()
    
    return {
        'tracker': tracker,
        'camera_estimator': camera_estimator,
        'team_assigner': team_assigner,
        'player_assigner': player_assigner,
        'view_transformer': view_transformer,
        'speed_distance_estimator': speed_distance_estimator
    }
