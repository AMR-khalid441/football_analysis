import itertools
import gc
import cv2
import numpy as np
import logging
from utils import bbox_utils

logger = logging.getLogger("football_analysis")

def process_video_frames(video_gen, first_frame, modules, config, out_writer):
    """
    Extract lines 135-291 from main.py
    Main frame processing loop with all detection, tracking, and annotation logic
    """
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
        frame_tracks = modules['tracker'].get_object_tracks([frame], read_from_stub=False, stub_path=None)
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
            cam_movements = modules['camera_estimator'].get_camera_movement([frame])
            camera_movement = cam_movements[0] if isinstance(cam_movements, (list, tuple)) and len(cam_movements) > 0 else [0, 0]
        except Exception as e:
            logger.warning(f"Camera movement estimation failed at frame {frame_num}: {e}")
            camera_movement = [0, 0]
        
        # adjust tracks if estimator supports it
        try:
            modules['camera_estimator'].add_adjust_positions_to_tracks(tracks, camera_movement)
        except (TypeError, AttributeError) as e:
            logger.debug(f"Position adjustment skipped: {e}")
            pass

        # Initialize team colors if not ready and enough players exist now
        if not hasattr(modules['team_assigner'], "kmeans") or modules['team_assigner'].kmeans is None:
            if len(players_frame) >= config['MIN_PLAYERS_FOR_KMEANS']:
                try:
                    modules['team_assigner'].assign_team_color(frame, players_frame)
                    logger.info(f"Team colors initialized at frame {frame_num} with {len(players_frame)} players.")
                except Exception as e:
                    logger.warning(f"Team color initialization failed at frame {frame_num}: {e}")

        # Team assignment (only if kmeans is initialized)
        if hasattr(modules['team_assigner'], "kmeans") and modules['team_assigner'].kmeans is not None:
            for player_id, player_data in players_frame.items():
                try:
                    # Use new hysteresis-aware assignment
                    team = modules['team_assigner'].get_player_team(frame, player_data['bbox'], player_id)
                    
                    tracks["players"][frame_num][player_id]['team'] = team
                    
                    # Get team color (handle -1 for unknown)
                    if team in modules['team_assigner'].team_colors:
                        tracks["players"][frame_num][player_id]['team_color'] = modules['team_assigner'].team_colors[team]
                    else:
                        tracks["players"][frame_num][player_id]['team_color'] = (0, 255, 0)  # Green for unknown
                        
                except Exception as e:
                    logger.debug(f"Team assignment failed for player {player_id}: {e}")
                    tracks["players"][frame_num][player_id]['team'] = -1
                    tracks["players"][frame_num][player_id]['team_color'] = (0, 255, 0)
        else:
            # Same as before: set all to -1 if kmeans not ready
            logger.debug("Team KMeans not initialized yet; skipping team assignment this frame.")
            for player_id in players_frame.keys():
                tracks["players"][frame_num][player_id]['team'] = -1
                tracks["players"][frame_num][player_id]['team_color'] = (0, 255, 0)

        # Ball assignment with timeout handling
        assigned_player = -1
        if ball_bbox:
            try:
                assigned_player = modules['player_assigner'].assign_ball_to_player(players_frame, ball_bbox)
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
            if ball_missing_count <= config['BALL_MAX_GAP_FRAMES'] and team_ball_control:
                team_ball_control.append(team_ball_control[-1])  # carry forward last team
            else:
                team_ball_control.append(-1)  # unknown team

        # Draw annotations
        annotated_frame = modules['tracker'].draw_annotations_single_frame(frame, frame_num, tracks, team_ball_control)
        # draw_camera_movement expects a list of frames and returns a list -> pass single-frame list and take first result
        cam_out = modules['camera_estimator'].draw_camera_movement([annotated_frame], camera_movement)
        if isinstance(cam_out, (list, tuple)):
            annotated_frame = cam_out[0] if len(cam_out) > 0 else annotated_frame
        else:
            annotated_frame = cam_out

        annotated_frame = modules['speed_distance_estimator'].draw_speed_and_distance(annotated_frame, tracks, frame_num)

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
    logger.info(f"Video saved successfully at {config['output_path']}")
