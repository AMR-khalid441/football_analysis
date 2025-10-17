import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame=None):
        # allow lazy initialization when frame is not yet available
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        # if a frame is provided, create feature mask now; otherwise defer until first frame
        if frame is not None:
            first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            mask_features = np.zeros_like(first_frame_grayscale)
            h, w = mask_features.shape
            # keep indices safe for different widths
            mask_features[:, 0:20] = 1
            mask_features[:, max(0, w-150):w] = 1
            self.features = dict(
                maxCorners = 100,
                qualityLevel = 0.3,
                minDistance =3,
                blockSize = 7,
                mask = mask_features
            )
        else:
            self.features = None

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust stored object positions by removing camera movement.
        - tracks: dict with lists per-frame, e.g. {"players":[frame0_dict, frame1_dict,...], "ball": [...]}
        - camera_movement_per_frame: either a single [dx,dy] or a list of [dx,dy] per frame
        This function will add/overwrite 'position_adjusted' in each track info dict.
        """
        # Determine number of frames from tracks (take max list length)
        num_frames = 0
        for obj_list in tracks.values():
            if isinstance(obj_list, (list, tuple)):
                num_frames = max(num_frames, len(obj_list))

        # Normalize camera_movement_per_frame to a list per-frame
        if camera_movement_per_frame is None:
            camera_movements = [[0, 0]] * num_frames
        elif (isinstance(camera_movement_per_frame, (list, tuple))
              and len(camera_movement_per_frame) == 2
              and not any(isinstance(x, (list, tuple)) and len(x) == 2 for x in camera_movement_per_frame)):
            # single [dx,dy] -> repeat for all frames
            camera_movements = [list(camera_movement_per_frame)] * num_frames
        else:
            camera_movements = list(camera_movement_per_frame)
            # pad if shorter than num_frames
            if len(camera_movements) < num_frames:
                camera_movements += [[0, 0]] * (num_frames - len(camera_movements))

        # Iterate and adjust
        for obj_name, obj_tracks in tracks.items():
            # obj_tracks expected to be list of per-frame dicts
            if not isinstance(obj_tracks, (list, tuple)):
                continue
            for frame_num, frame_tracks in enumerate(obj_tracks):
                if not isinstance(frame_tracks, dict):
                    continue
                cam_move = camera_movements[frame_num] if frame_num < len(camera_movements) else [0, 0]
                dx, dy = cam_move[0], cam_move[1]
                for track_id, track_info in frame_tracks.items():
                    if not isinstance(track_info, dict):
                        continue
                    # Prefer explicit 'position' if present, otherwise derive from bbox
                    if 'position' in track_info and track_info['position'] is not None:
                        pos = track_info['position']
                    elif 'bbox' in track_info and track_info['bbox']:
                        bbox = track_info['bbox']
                        try:
                            x1, y1, x2, y2 = map(float, bbox[:4])
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            pos = (cx, cy)
                        except Exception:
                            # fallback to (0,0) if bbox malformed
                            pos = (0.0, 0.0)
                    else:
                        # nothing to adjust
                        continue

                    # Adjust position by camera movement (subtract camera motion)
                    pos_adj = (pos[0] - dx, pos[1] - dy)
                    track_info['position_adjusted'] = pos_adj
                    


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # avoid shared-list reference bug
        camera_movement = [[0,0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)

        # lazy-init features if constructor didn't receive a frame
        if self.features is None:
            mask_features = np.zeros_like(old_gray)
            h, w = mask_features.shape
            mask_features[:, 0:20] = 1
            mask_features[:, max(0, w-150):w] = 1
            self.features = dict(
                maxCorners = 100,
                qualityLevel = 0.3,
                minDistance =3,
                blockSize = 7,
                mask = mask_features
            )

        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        if old_features is None:
            # no reliable features found — return zeros (or you can raise)
            print("Warning: no features detected on first frame for camera motion estimation.")
            return camera_movement

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            # guard against calcOpticalFlowPyrLK failures
            if new_features is None or status is None:
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if old_features is None:
                    # skip this frame if no features
                    continue
                else:
                    continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            # only use good points where status == 1
            good_new = new_features[status.ravel()==1]
            good_old = old_features[status.ravel()==1]

            for new_pt, old_pt in zip(good_new, good_old):
                new_features_point = new_pt.ravel()
                old_features_point = old_pt.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        # Normalize camera_movement_per_frame to a list of [dx,dy] for each frame
        n = len(frames)
        camera_movements = None
        if camera_movement_per_frame is None:
            camera_movements = [[0, 0]] * n
        elif isinstance(camera_movement_per_frame, (list, tuple)) and len(camera_movement_per_frame) == 2 and isinstance(camera_movement_per_frame[0], (int, float)):
            # single [dx,dy] provided -> repeat for all frames
            camera_movements = [list(camera_movement_per_frame)] * n
        else:
            camera_movements = list(camera_movement_per_frame)
            # if flat list of numbers (e.g. [dx1,dy1, dx2,dy2, ...]) pair them
            if len(camera_movements) >= 1 and isinstance(camera_movements[0], (int, float)):
                if len(camera_movements) >= 2 * n:
                    paired = []
                    for i in range(n):
                        paired.append([camera_movements[2 * i], camera_movements[2 * i + 1]])
                    camera_movements = paired
                else:
                    # not enough data: pad/truncate to n entries
                    paired = []
                    for i in range(n):
                        if 2 * i + 1 < len(camera_movements):
                            paired.append([camera_movements[2 * i], camera_movements[2 * i + 1]])
                        elif i < len(camera_movements):
                            paired.append([camera_movements[i], 0])
                        else:
                            paired.append([0, 0])
                    camera_movements = paired

        # Draw movement indicator on each frame
        for frame_num, frame in enumerate(frames):
            out = frame.copy()
            dx, dy = [0, 0]
            if frame_num < len(camera_movements):
                mv = camera_movements[frame_num]
                # ensure mv is pair-like
                if isinstance(mv, (list, tuple)) and len(mv) >= 2:
                    dx, dy = float(mv[0]), float(mv[1])
                elif isinstance(mv, (int, float)):
                    dx, dy = float(mv), 0.0
            h, w = out.shape[:2]
            cx, cy = w // 2, h // 2
            end_pt = (int(cx + dx), int(cy + dy))
            try:
                cv2.arrowedLine(out, (cx, cy), end_pt, (0, 0, 255), 2, tipLength=0.2)
                cv2.putText(out, f"cam_mv: [{dx:.1f},{dy:.1f}]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            except Exception:
                pass
            output_frames.append(out)

        return output_frames