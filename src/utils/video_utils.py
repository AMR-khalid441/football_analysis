import cv2
import os

def read_video(input_path):
    """
    Memory-efficient video reader.
    Instead of loading all frames, it yields one frame at a time.
    Usage:
        for frame in read_video(path):
            # process frame
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame  # yield instead of storing in a list

    cap.release()


def save_video(frame_iterator, output_path, fps=24, frame_size=None):
    """
    Save video frames efficiently.
    frame_iterator: iterator or generator of frames
    fps: frames per second
    frame_size: (width, height). If None, will take from first frame
    """
    first_frame = next(frame_iterator, None)
    if first_frame is None:
        raise ValueError("No frames to save.")

    if frame_size is None:
        height, width = first_frame.shape[:2]
        frame_size = (width, height)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Write first frame
    out.write(first_frame)

    # Write remaining frames
    for frame in frame_iterator:
        out.write(frame)

    out.release()
    print(f"✅ Video saved successfully at {output_path}")
