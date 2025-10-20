import os
import cv2
import numpy as np
import itertools
import logging
import gc
from utils import read_video, save_video, bbox_utils
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("football_analysis")

def main():
    # -------------------------------
    # Configuration constants
    # -------------------------------
    MIN_PLAYERS_FOR_KMEANS = 6
    BALL_MAX_GAP_FRAMES = 15
    
    # -------------------------------
    # Base directory
    # -------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------
    # Paths
    # -------------------------------
    input_path = os.path.join(script_dir, "input_videos", "CV_Task.mkv")
    output_path = os.path.join(script_dir, "output_videos", "output_video.avi")
    model_path = os.path.join(script_dir, "models", "best.pt")
    track_stub_path = os.path.join(script_dir, "stubs", "track_stubs.pkl")
    camera_stub_path = os.path.join(script_dir, "stubs", "camera_movement_stub.pkl")

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

    # -------------------------------
    # Initialize modules
    # -------------------------------
    tracker = Tracker(model_path)
    camera_estimator = CameraMovementEstimator(None)  # frame will be set later
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    view_transformer = ViewTransformer()
    speed_distance_estimator = SpeedAndDistance_Estimator()

    # -------------------------------
    # Video reader generator
    # -------------------------------
    video_gen = read_video(input_path)
    first_frame = next(video_gen, None)
    if first_frame is None:
        raise ValueError("No frames to process")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

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
    if len(valid_initial_players) >= MIN_PLAYERS_FOR_KMEANS:
        try:
            team_assigner.assign_team_color(first_frame, valid_initial_players)
            logger.info(f"Team colors initialized from first frame with {len(valid_initial_players)} players.")
        except Exception as e:
            logger.warning(f"Team color initialization failed: {e}")
    else:
        logger.warning(f"Insufficient players to initialize team colors ({len(valid_initial_players)}/{MIN_PLAYERS_FOR_KMEANS}); will try again later.")

    # Storage for tracks and ball control
    tracks = {"players": [], "ball": []}
    team_ball_control = []
    ball_missing_count = 0

    # -------------------------------
    # Process frames one by one
    # -------------------------------
    frame_num = 0
    # iterate without loading all frames into memory
    for frame in itertools.chain([first_frame], video_gen):
        # Tracker: detect objects for this single frame (Tracker expects list input)
        frame_tracks = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        # normalize players/ball for single-frame return (could be list or dict)
        players_per_frame = frame_tracks.get("players", [])
        if isinstance(players_per_frame, list) and len(players_per_frame) > 0:
            players_frame = players_per_frame[0]
        elif isinstance(players_per_frame, dict):
            players_frame = players_per_frame
        else:
            players_frame = {}
        ball_per_frame = frame_tracks.get("ball", [])
        if isinstance(ball_per_frame, list) and len(ball_per_frame) > 0:
            ball_frame = ball_per_frame[0]
        elif isinstance(ball_per_frame, dict):
            ball_frame = ball_per_frame
        else:
            ball_frame = {}

        # Filter invalid player bboxes
        filtered_players = {}
        for pid, pdata in players_frame.items():
            bbox = pdata.get('bbox')
            if bbox_utils.is_valid_bbox(bbox, frame.shape):
                filtered_players[pid] = pdata
            else:
                logger.debug(f"Invalid player bbox skipped (id={pid}): {bbox}")
        players_frame = filtered_players

        # Validate ball bbox
        ball_bbox = None
        if isinstance(ball_frame, dict):
            if 1 in ball_frame and bbox_utils.is_valid_bbox(ball_frame[1].get('bbox'), frame.shape):
                ball_bbox = ball_frame[1]['bbox']
            else:
                logger.debug(f"Invalid ball bbox skipped: {ball_frame.get(1, {}).get('bbox')}")

        tracks["players"].append(players_frame)
        tracks["ball"].append(ball_frame)

        # Camera movement for this single frame (returns list of movements for input frames)
        try:
            cam_movements = camera_estimator.get_camera_movement([frame])
            camera_movement = cam_movements[0] if isinstance(cam_movements, (list, tuple)) and len(cam_movements) > 0 else [0, 0]
        except Exception as e:
            logger.warning(f"Camera movement estimation failed at frame {frame_num}: {e}")
            camera_movement = [0, 0]
        
        # adjust tracks if estimator supports it
        try:
            camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        except (TypeError, AttributeError) as e:
            logger.debug(f"Position adjustment skipped: {e}")
            pass

        # Initialize team colors if not ready and enough players exist now
        if not hasattr(team_assigner, "kmeans") or team_assigner.kmeans is None:
            if len(players_frame) >= MIN_PLAYERS_FOR_KMEANS:
                try:
                    team_assigner.assign_team_color(frame, players_frame)
                    logger.info(f"Team colors initialized at frame {frame_num} with {len(players_frame)} players.")
                except Exception as e:
                    logger.warning(f"Team color initialization failed at frame {frame_num}: {e}")

        # Team assignment (only if kmeans is initialized)
        if hasattr(team_assigner, "kmeans") and team_assigner.kmeans is not None:
            for player_id, player_data in players_frame.items():
                try:
                    team = team_assigner.get_player_team(frame, player_data['bbox'], player_id)
                    tracks["players"][frame_num][player_id]['team'] = team
                    tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 255, 0))
                except Exception as e:
                    logger.debug(f"Team assignment failed for player {player_id}: {e}")
                    tracks["players"][frame_num][player_id]['team'] = -1
                    tracks["players"][frame_num][player_id]['team_color'] = (0, 255, 0)
        else:
            logger.debug("Team KMeans not initialized yet; skipping team assignment this frame.")
            for player_id, player_data in players_frame.items():
                tracks["players"][frame_num][player_id]['team'] = -1
                tracks["players"][frame_num][player_id]['team_color'] = (0, 255, 0)

        # Ball assignment with timeout handling
        assigned_player = -1
        if ball_bbox:
            try:
                assigned_player = player_assigner.assign_ball_to_player(players_frame, ball_bbox)
            except Exception as e:
                logger.debug(f"Ball assignment error: {e}")

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]['has_ball'] = True
            if 'team' in players_frame[assigned_player]:
                team_ball_control.append(players_frame[assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
            ball_missing_count = 0
        else:
            ball_missing_count += 1
            if ball_missing_count <= BALL_MAX_GAP_FRAMES and team_ball_control:
                team_ball_control.append(team_ball_control[-1])  # carry forward last team
            else:
                team_ball_control.append(-1)  # unknown team

        # Draw annotations
        annotated_frame = tracker.draw_annotations_single_frame(frame, frame_num, tracks, team_ball_control)
        # draw_camera_movement expects a list of frames and returns a list -> pass single-frame list and take first result
        cam_out = camera_estimator.draw_camera_movement([annotated_frame], camera_movement)
        if isinstance(cam_out, (list, tuple)):
            annotated_frame = cam_out[0] if len(cam_out) > 0 else annotated_frame
        else:
            annotated_frame = cam_out

        annotated_frame = speed_distance_estimator.draw_speed_and_distance(annotated_frame, tracks, frame_num)

        # Ensure annotated_frame is a proper numpy uint8 BGR image for VideoWriter
        if not isinstance(annotated_frame, np.ndarray):
            annotated_frame = np.array(annotated_frame)
        if annotated_frame.dtype != np.uint8:
            try:
                annotated_frame = annotated_frame.astype(np.uint8)
            except Exception:
                raise TypeError("annotated_frame cannot be converted to uint8 array for VideoWriter")
        # if single-channel, convert to BGR
        if annotated_frame.ndim == 2:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)
        # if has alpha, drop it
        if annotated_frame.shape[2] == 4:
            annotated_frame = annotated_frame[:, :, :3]

        out_writer.write(annotated_frame)
        frame_num += 1
        
        # Force garbage collection every 10 frames to prevent memory accumulation
        if frame_num % 10 == 0:
            gc.collect()

    out_writer.release()
    logger.info(f"Video saved successfully at {output_path}")

if __name__ == "__main__":
    main()
