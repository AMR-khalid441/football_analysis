import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
        # Add smoothing infrastructure
        self.smoothed_speeds = {}  # track_id -> last_smoothed_speed
        self.smoothing_alpha = 0.3  # Configurable smoothing parameter (0-1)
    
    def smooth_speed_exponential(self, track_id, new_speed):
        """
        Apply exponential smoothing to speed measurements.
        
        Formula: smoothed = α * new + (1-α) * previous
        Lower α = more smoothing, Higher α = more responsive
        
        Args:
            track_id: Unique identifier for the track
            new_speed: Raw speed value from calculation (km/h)
            
        Returns:
            Smoothed speed value (km/h)
        """
        if track_id not in self.smoothed_speeds:
            # First speed for this track - no smoothing needed
            self.smoothed_speeds[track_id] = new_speed
        else:
            # Apply exponential smoothing: α * new + (1-α) * old
            self.smoothed_speeds[track_id] = (
                self.smoothing_alpha * new_speed + 
                (1 - self.smoothing_alpha) * self.smoothed_speeds[track_id]
            )
        
        return self.smoothed_speeds[track_id]
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        smoothed_speed = self.smooth_speed_exponential(track_id, speed_km_per_hour)
                        tracks[object][frame_num_batch][track_id]['speed'] = smoothed_speed
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self, annotated_frame, tracks, frame_num=None):
        """
        Backwards-compatible: accepts optional frame_num.
        If the original implementation didn't expect frame_num, it will be ignored.
        """
        output_frame = annotated_frame.copy()
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            for _, track_info in object_tracks[frame_num].items():
               if "speed" in track_info:
                   speed = track_info.get('speed',None)
                   distance = track_info.get('distance',None)
                   if speed is None or distance is None:
                       continue
                       
                   bbox = track_info['bbox']
                   position = get_foot_position(bbox)
                   position = list(position)
                   position[1]+=40

                   position = tuple(map(int,position))
                   cv2.putText(output_frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                   cv2.putText(output_frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        return output_frame