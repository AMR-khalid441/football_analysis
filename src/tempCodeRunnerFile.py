import os
import cv2
import numpy as np
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
    for frame in (f for f in [first_frame] + list(video_gen)):
        # Tracker: detect objects and update tracks
        frame_tracks = tracker.get_object_tracks(frame, frame_num)  # you may need a frame-by-frame method
        tracks["players"].append(frame_tracks.get("players", {}))
        tracks["ball"].append(frame_tracks.get("ball", {}))

        # Camera movement
        camera_movement = camera_estimator.get_camera_movement_for_frame(frame, frame_num)
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement, frame_num)

        # View transformation
        view_transformer.add_transformed_position_to_tracks(tracks, frame_num)

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
        annotated_frame = camera_estimator.draw_camera_movement_single_frame(annotated_frame, camera_movement)
        speed_distance_estimator.draw_speed_and_distance_single_frame(annotated_frame, tracks, frame_num)

        # Write frame
        out_writer.write(annotated_frame)
        frame_num += 1

    out_writer.release()
    print(f"✅ Video saved successfully at {output_path}")

if __name__ == "__main__":
    main()
