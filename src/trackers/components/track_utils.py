import pandas as pd
import sys
sys.path.append('../../')
from utils.bbox_utils import get_center_of_bbox, get_foot_position

def add_position_to_tracks(tracks):
    """
    Extract lines 27-36 from tracker.py
    Add position information to tracks
    """
    for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object][frame_num][track_id]['position'] = position

def interpolate_ball_positions(ball_positions):
    """
    Extract lines 38-48 from tracker.py
    Interpolate missing ball positions
    """
    ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

    # Interpolate missing values
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()

    ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    return ball_positions
