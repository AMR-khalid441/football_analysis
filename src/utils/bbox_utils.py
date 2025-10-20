import numpy as np

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def is_valid_bbox(bbox, frame_shape=None):
    """
    Validate bounding box format and bounds.
    
    Args:
        bbox: List/tuple of [x1, y1, x2, y2] or None
        frame_shape: Optional (height, width, channels) for bounds checking
        
    Returns:
        bool: True if bbox is valid, False otherwise
    """
    try:
        if bbox is None:
            return False
        
        if not isinstance(bbox, (list, tuple, np.ndarray)):
            return False
            
        if len(bbox) < 4:
            return False
            
        x1, y1, x2, y2 = bbox[:4]
        
        # Check for None or NaN values
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (x1, y1, x2, y2)):
            return False
            
        # Check bbox dimensions are valid
        if not (x2 > x1 and y2 > y1):
            return False
            
        # Optional frame bounds checking
        if frame_shape is not None:
            h, w = frame_shape[:2]
            # Allow slight overflow but reject absurd boxes
            if x1 < -w*0.1 or y1 < -h*0.1 or x2 > w*1.1 or y2 > h*1.1:
                return False
                
        return True
        
    except Exception:
        return False