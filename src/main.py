import os
import cv2
import numpy as np
import itertools
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # -------------------------------
    # Base directory
    # -------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------
    # Paths
    # -------------------------------
    input_path = os.path.join(script_dir, "input_videos", "videone.mp4")
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
        print(f"⚠ Warning: Track stub not found: {track_stub_path} (will regenerate if needed)")
    if not os.path.exists(camera_stub_path):
        print(f"⚠ Warning: Camera movement stub not found: {camera_stub_path} (will regenerate if needed)")

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
    # call assign_team_color with a dict (or empty dict)
    team_assigner.assign_team_color(first_frame, initial_players)

    # Storage for tracks and ball control
    tracks = {"players": [], "ball": []}
    team_ball_control = []

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

        tracks["players"].append(players_frame)
        tracks["ball"].append(ball_frame)

        # Camera movement for this single frame (returns list of movements for input frames)
        cam_movements = camera_estimator.get_camera_movement([frame])
        camera_movement = cam_movements[0] if isinstance(cam_movements, (list, tuple)) and len(cam_movements) > 0 else [0, 0]
        # adjust tracks if estimator supports it
        try:
            camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        except TypeError:
            # some versions may expect different args — ignore if not applicable
            pass

        # Team assignment
        players_frame = tracks["players"][frame_num]
        for player_id, player_data in players_frame.items():
            team = team_assigner.get_player_team(frame, player_data['bbox'], player_id)
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Ball assignment
        ball_frame = tracks["ball"][frame_num]
        ball_bbox = ball_frame[1]['bbox'] if 1 in ball_frame else None
        assigned_player = player_assigner.assign_ball_to_player(players_frame, ball_bbox) if ball_bbox else -1
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(players_frame[assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

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

    out_writer.release()
    print(f"✅ Video saved successfully at {output_path}")

if __name__ == "__main__":
    main()
