import pickle
import cv2
import numpy as np
import os
import sys
import logging
import gc
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

logger = logging.getLogger(__name__)

class OpticalFlowConfig:
    def __init__(self, frame=None):
        # allow lazy initialization when frame is not yet available
        self.minimum_distance = 5
        
        # Memory optimization parameters
        self.scale = 0.5  # Compute flow at half resolution
        self.reinit_interval = 50  # Refresh features every 50 frames
        self.frame_count = 0

        # Optimized Lucas-Kanade parameters for memory efficiency
        self.lk_params = dict(
            winSize = (13, 13),  # Reduced from (15,15)
            maxLevel = 1,        # Reduced from 2 (shallower pyramid)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
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
                maxCorners = 150,      # More features but better quality
                qualityLevel = 0.02,    # More strict selection
                minDistance = 8,       # Increased spacing (fewer overlapping)
                blockSize = 7,
                mask = mask_features
            )
        else:
            self.features = None
