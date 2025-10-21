<!-- d25977f9-d20b-4aee-8a66-9152704498fe 334e72ad-2c58-4244-9898-001e00e45927 -->
# Refactor tracker.py into Modular Components

## Overview

Extract the 337-line tracker.py into focused modules within the `trackers/` folder. Keep the exact same code logic, just organized by single responsibility.

## File Structure

```
src/trackers/
├── __init__.py (update imports)
├── tracker.py (new simplified version)
├── components/
│   ├── __init__.py
│   ├── model_manager.py      # Lines 13-25
│   ├── detection_processor.py # Lines 50-69
│   ├── tracking_engine.py     # Lines 71-165
│   ├── visualizer.py          # Lines 167-337
│   └── track_utils.py         # Lines 27-48
```

## Implementation Steps

### 1. Create components folder structure

- Create `src/trackers/components/` directory
- Create `src/trackers/components/__init__.py`

### 2. Create model_manager.py

Extract lines 13-25 from tracker.py into `ModelManager` class:

- Constructor loads YOLO model with error handling
- Try to load custom model from model_path
- Catch exceptions and print error message
- Fallback to smaller model (yolov5su.pt) if custom fails
- Initialize ByteTrack tracker
- Store self.model and self.tracker as instance variables

### 3. Create detection_processor.py

Extract lines 50-69 from tracker.py into `DetectionProcessor` class:

- Constructor accepts model reference
- `detect_frames()` method processes frames in batches
- Batch size = 1 to prevent OOM
- Garbage collection before each prediction
- Use conf=0.1 and verbose=False for model.predict()
- Handle RuntimeError for memory errors
- Return list of detections (or None for failed frames)

### 4. Create tracking_engine.py

Extract lines 71-165 from tracker.py into `TrackingEngine` class:

- Constructor accepts model and tracker references
- `get_object_tracks()` method - main tracking logic:
  - Load from stub if requested (lines 73-76)
  - Call detect_frames() for detections (line 78)
  - Initialize tracks dict (lines 80-84)
  - Loop through detections per frame (lines 86-159):
    - Handle None detections (lines 88-92)
    - Get class names and inverse mapping (lines 94-95)
    - Print available classes on first frame (lines 98-99)
    - Convert to supervision format (line 102)
    - Map class IDs for player/referee/ball (lines 107-127)
    - Convert goalkeeper to player (lines 130-133)
    - Update tracks with ByteTrack (line 136)
    - Append empty dicts for each object type (lines 138-140)
    - Fill tracks for players and referees (lines 142-152)
    - Fill tracks for ball (lines 154-159)
  - Save to stub if path provided (lines 161-163)
  - Return tracks

### 5. Create visualizer.py

Extract lines 167-337 from tracker.py into `Visualizer` class:

- `draw_ellipse()` method (lines 167-212):
  - Extract bbox coordinates and center
  - Draw ellipse at player feet
  - Draw rectangle with track ID
  - Draw ID text inside rectangle
- `draw_triangle()` method (lines 214-226):
  - Calculate triangle points above bbox
  - Draw filled triangle with border
- `draw_annotations_single_frame()` method (lines 228-304):
  - Copy frame
  - Extract players for current frame
  - Normalize list/dict structures
  - Draw each player with bbox, label, ellipse
  - Extract ball for current frame
  - Draw ball as circle
  - Try to draw team ball control widget
- `draw_annotations()` method (lines 306-337):
  - Batch process video frames
  - Get player/ball/referee dicts per frame
  - Draw ellipses for players with team colors
  - Draw triangles for ball possession
  - Draw ellipses for referees
  - Draw triangles for ball
  - Draw team ball control widget
  - Return annotated frames list

### 6. Create track_utils.py

Extract lines 27-48 from tracker.py into utility functions:

- `add_position_to_tracks()` function (lines 27-36):
  - Loop through objects and tracks
  - Calculate position (center for ball, foot for players)
  - Add position to track info
- `interpolate_ball_positions()` function (lines 38-48):
  - Extract ball bboxes from tracks
  - Create pandas DataFrame
  - Interpolate missing values
  - Backfill remaining gaps
  - Convert back to track format
  - Return interpolated positions

### 7. Refactor tracker.py

Replace current tracker.py with Tracker class that uses components:

- Import all component classes
- Constructor initializes all components:
  - model_manager = ModelManager(model_path)
  - detection_processor = DetectionProcessor(model_manager.model)
  - tracking_engine = TrackingEngine(model_manager.model, model_manager.tracker)
  - visualizer = Visualizer()
- Delegate methods to components:
  - `add_position_to_tracks()` → calls track_utils function
  - `interpolate_ball_positions()` → calls track_utils function
  - `detect_frames()` → calls detection_processor.detect_frames()
  - `get_object_tracks()` → calls tracking_engine.get_object_tracks()
  - `draw_ellipse()` → calls visualizer.draw_ellipse()
  - `draw_triangle()` → calls visualizer.draw_triangle()
  - `draw_annotations_single_frame()` → calls visualizer.draw_annotations_single_frame()
  - `draw_annotations()` → calls visualizer.draw_annotations()

### 8. Update **init**.py

Update `src/trackers/__init__.py` to export Tracker class

## Key Principles

- Keep exact same code logic (no functional changes)
- Each component has single responsibility
- Maintain all method signatures for backward compatibility
- Preserve all imports in appropriate modules
- Keep all error handling and logging
- Maintain same data structures and return values