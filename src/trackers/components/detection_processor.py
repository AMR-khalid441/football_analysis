class DetectionProcessor:
    def __init__(self, model):
        """
        Extract lines 50-69 from tracker.py
        Detection processing with memory management
        """
        self.model = model
    
    def detect_frames(self, frames):
        """
        Extract lines 50-69 from tracker.py
        Process frames in batches with memory optimization
        """
        batch_size = 1  # Reduced from 20 to prevent OOM
        detections = [] 
        for i in range(0, len(frames), batch_size):
            try:
                # Force garbage collection before each prediction
                import gc
                gc.collect()
                
                # Use lower confidence to reduce memory usage
                detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, verbose=False)
                detections += detections_batch
            except RuntimeError as e:
                if "not enough memory" in str(e):
                    print(f"Memory error in detection, skipping frame {i}: {e}")
                    # Add empty detection to maintain frame count
                    detections.append(None)
                else:
                    raise e
        return detections
