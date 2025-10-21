import logging
from utils import bbox_utils

logger = logging.getLogger("football_analysis")

def initialize_team_colors(tracker, team_assigner, camera_estimator, first_frame, config):
    """
   
    Initialize team colors using K-Means clustering on first frame
    """
    # Initialize trackers with first frame if needed
    camera_estimator.first_frame = first_frame

    # get initial detections for the first frame so team colors can be determined
    # Tracker.get_object_tracks expects a list of frames and returns per-frame lists
    initial_tracks = tracker.get_object_tracks([first_frame], read_from_stub=False, stub_path=None)
    initial_players = {}
    if isinstance(initial_tracks, dict):
        players_per_frame = initial_tracks.get("players", [])
        if isinstance(players_per_frame, list) and len(players_per_frame) > 0:
            frame0_players = players_per_frame[0]
            # normalize to dict player_id -> { "bbox": ... }
            if isinstance(frame0_players, dict):
                initial_players = frame0_players
            elif isinstance(frame0_players, list):
                # try to convert list of detections/bboxes to dict using indices or 'id' field
                initial_players = {}
                for idx, det in enumerate(frame0_players):
                    if isinstance(det, dict) and "id" in det:
                        pid = det.get("id", idx)
                        initial_players[pid] = det
                    elif isinstance(det, (list, tuple)) and len(det) >= 4:
                        initial_players[idx] = {"bbox": det}
                    else:
                        initial_players[idx] = {"bbox": det}
    
    # Filter valid players for team color initialization
    valid_initial_players = {}
    for pid, det in initial_players.items():
        bbox = det.get("bbox")
        if bbox_utils.is_valid_bbox(bbox, first_frame.shape):
            valid_initial_players[pid] = det
        else:
            logger.debug(f"Invalid initial player bbox skipped (id={pid}): {bbox}")
    
    # Initialize team colors if enough valid players
    if len(valid_initial_players) >= config['MIN_PLAYERS_FOR_KMEANS']:
        try:
            team_assigner.assign_team_color(first_frame, valid_initial_players)
            logger.info(f"Team colors initialized from first frame with {len(valid_initial_players)} players.")
        except Exception as e:
            logger.warning(f"Team color initialization failed: {e}")
    else:
        logger.warning(f"Insufficient players to initialize team colors ({len(valid_initial_players)}/{config['MIN_PLAYERS_FOR_KMEANS']}); will try again later.")
