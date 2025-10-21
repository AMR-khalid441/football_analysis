class PositionAdjuster:
    def adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
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
