import pickle
import cv2
import numpy as np
import os
import logging
import gc

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, config):
        self.config = config

    def detect_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # avoid shared-list reference bug
        camera_movement = [[0,0] for _ in range(len(frames))]

        # Downscale frames for memory-efficient optical flow
        old_gray_full = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_gray = cv2.resize(old_gray_full, None, fx=self.config.scale, fy=self.config.scale, 
                             interpolation=cv2.INTER_AREA)

        # lazy-init features if constructor didn't receive a frame
        if self.config.features is None:
            mask_features = np.zeros_like(old_gray)
            h, w = mask_features.shape
            mask_features[:, 0:20] = 1
            mask_features[:, max(0, w-150):w] = 1
            self.config.features = dict(
                maxCorners = 150,      # More features but better quality
                qualityLevel = 0.02,    # More strict selection
                minDistance = 8,       # Increased spacing (fewer overlapping)
                blockSize = 7,
                mask = mask_features
            )

        old_features = cv2.goodFeaturesToTrack(old_gray,**self.config.features)
        if old_features is None:
            # no reliable features found — return zeros (or you can raise)
            print("Warning: no features detected on first frame for camera motion estimation.")
            return camera_movement

        for frame_num in range(1,len(frames)):
            try:
                # Downscale current frame for optical flow
                frame_gray_full = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.resize(frame_gray_full, None, fx=self.config.scale, fy=self.config.scale, 
                                       interpolation=cv2.INTER_AREA)
                
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.config.lk_params)
                
            except cv2.error as e:
                if 'memory' in str(e).lower():
                    logger.warning(f"Memory error at frame {frame_num}, using zero motion")
                    camera_movement[frame_num] = [0, 0]
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Re-initialize features
                    old_gray = cv2.resize(frame_gray_full, None, fx=self.config.scale, fy=self.config.scale, 
                                         interpolation=cv2.INTER_AREA)
                    old_features = cv2.goodFeaturesToTrack(old_gray, **self.config.features)
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error at frame {frame_num}: {e}")
                camera_movement[frame_num] = [0, 0]
                continue

            # guard against calcOpticalFlowPyrLK failures
            if new_features is None or status is None:
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.config.features)
                if old_features is None:
                    # skip this frame if no features
                    continue
                else:
                    continue

            # Collect all valid movements for median consensus
            movements = []
            
            # only use good points where status == 1
            good_new = new_features[status.ravel()==1]
            good_old = old_features[status.ravel()==1]
            
            logger.debug(f"Frame {frame_num}: {len(good_new)} features tracked")

            for new_pt, old_pt in zip(good_new, good_old):
                new_features_point = new_pt.ravel()
                old_features_point = old_pt.ravel()
                
                dx = new_features_point[0] - old_features_point[0]
                dy = new_features_point[1] - old_features_point[1]
                distance = (dx*dx + dy*dy) ** 0.5
                
                if distance > self.config.minimum_distance * self.config.scale:
                    movements.append([dx, dy])
            
            # Use median consensus (robust to outliers)
            if len(movements) > 0:
                movements_array = np.array(movements)
                camera_movement_x = float(np.median(movements_array[:, 0]) / self.config.scale)
                camera_movement_y = float(np.median(movements_array[:, 1]) / self.config.scale)
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                
                logger.debug(f"Frame {frame_num}: motion=({camera_movement_x:.1f}, {camera_movement_y:.1f}) from {len(movements)} points")
            
            # Periodic feature re-initialization to prevent drift
            self.config.frame_count += 1
            
            # Re-initialize features periodically or when too few remain
            if self.config.frame_count % self.config.reinit_interval == 0 or len(good_new) < 20:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.config.features)
                logger.debug(f"Frame {frame_num}: Re-initialized features")
            else:
                old_features = good_new.reshape(-1, 1, 2)

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
