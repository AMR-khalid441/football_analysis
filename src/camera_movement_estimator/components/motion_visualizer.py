import cv2

class MotionVisualizer:
    def draw_camera_movement(self, frames, camera_movement_per_frame):
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
