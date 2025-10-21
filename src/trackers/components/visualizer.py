import cv2
import numpy as np
import sys
sys.path.append('../../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width

class Visualizer:
    def __init__(self):
        """
        Extract lines 167-337 from tracker.py
        Visualization and drawing functionality
        """
        pass
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Extract lines 167-212 from tracker.py
        Draw ellipse at player feet with track ID
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Extract lines 214-226 from tracker.py
        Draw triangle above bbox
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations_single_frame(self, frame, frame_num, tracks, team_ball_control):
        """
        Extract lines 228-304 from tracker.py
        Draw annotations for a single frame
        """
        out = frame.copy()

        # get players for this frame
        players_list = tracks.get("players", [])
        if isinstance(players_list, list):
            players_frame = players_list[frame_num] if frame_num < len(players_list) else {}
        elif isinstance(players_list, dict):
            players_frame = players_list
        else:
            players_frame = {}

        # normalize list-of-detections -> dict
        if isinstance(players_frame, list):
            players_dict = {}
            for i, det in enumerate(players_frame):
                if isinstance(det, dict) and "id" in det:
                    pid = det["id"]
                    players_dict[pid] = det
                elif isinstance(det, (list, tuple)) and len(det) >= 4:
                    players_dict[i] = {"bbox": det}
                else:
                    players_dict[i] = {"bbox": det}
        elif isinstance(players_frame, dict):
            players_dict = players_frame
        else:
            players_dict = {}

        # draw players
        for pid, pdata in players_dict.items():
            bbox = pdata.get("bbox")
            color = tuple(map(int, pdata.get("team_color", (0, 255, 0))))
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{pid}"
                cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # optional ellipse under player
                try:
                    self.draw_ellipse(out, bbox, color, track_id=pid)
                except Exception:
                    pass

        # draw ball if present
        ball_list = tracks.get("ball", [])
        if isinstance(ball_list, list):
            ball_frame = ball_list[frame_num] if frame_num < len(ball_list) else {}
        else:
            ball_frame = ball_list

        if isinstance(ball_frame, dict):
            # expected format: {id: {"bbox": [...]}, ...} often ball id == 1
            for bid, binfo in ball_frame.items():
                bb = binfo.get("bbox")
                if bb and len(bb) >= 4:
                    x1, y1, x2, y2 = map(int, bb[:4])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(out, (cx, cy), 8, (0, 165, 255), -1)
        elif isinstance(ball_frame, (list, tuple)) and len(ball_frame) >= 4:
            x1, y1, x2, y2 = map(int, ball_frame[:4])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(out, (cx, cy), 8, (0, 165, 255), -1)

        # draw team ball control widget if available
        try:
            self.draw_team_ball_control(out, frame_num, team_ball_control)
        except Exception:
            pass

        return out

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Extract lines 306-337 from tracker.py
        Batch process video frames with annotations
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
