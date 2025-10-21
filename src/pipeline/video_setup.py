import cv2
from utils import read_video

def setup_video_io(input_path, output_path):
    """
    Extract lines 78-88 from main.py
    Setup video input/output with first frame validation
    """
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
    
    return video_gen, first_frame, out_writer
