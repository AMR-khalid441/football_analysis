# 🔧 Technical Documentation

> **Comprehensive Technical Reference for Football Analysis System**

## 📋 Table of Contents

1. [Project Structure](#project-structure)
2. [Main Pipeline](#main-pipeline)
3. [Pipeline Layer](#pipeline-layer)
4. [Core Modules](#core-modules)
5. [Component Architecture](#component-architecture)
6. [Utility Functions](#utility-functions)
7. [Configuration System](#configuration-system)
8. [Error Handling](#error-handling)
9. [Performance Optimizations](#performance-optimizations)

---

## 📁 Project Structure

```
football_analysis/
├── src/                                    # 🎯 Source Code Directory
│   ├── main.py                            # 🚀 Main Pipeline Orchestrator
│   ├── pipeline/                          # 🔄 Pipeline Processing Layer
│   │   ├── __init__.py
│   │   ├── config_manager.py             # ⚙️ Configuration Management
│   │   ├── module_initializer.py         # 🚀 Module Initialization
│   │   ├── video_setup.py                # 📹 Video I/O Management
│   │   ├── team_initializer.py           # 🎨 Team Color Detection
│   │   └── frame_processor.py            # 🎬 Frame Processing Loop
│   ├── trackers/                          # 🎯 Object Detection & Tracking
│   │   ├── __init__.py
│   │   ├── tracker.py                    # 🎯 Main Tracker Orchestrator
│   │   └── components/                    # 🧩 Tracker Components
│   │       ├── __init__.py
│   │       ├── model_manager.py          # 🤖 YOLO Model Management
│   │       ├── detection_processor.py    # 🔍 Detection Processing
│   │       ├── tracking_engine.py        # 🚀 ByteTrack Engine
│   │       ├── visualizer.py             # 🎨 Visualization
│   │       └── track_utils.py            # 🛠️ Utility Functions
│   ├── camera_movement_estimator/         # 📹 Camera Motion Compensation
│   │   ├── __init__.py
│   │   ├── camera_movement_estimator.py  # 📹 Main Estimator
│   │   └── components/                    # 🧩 Motion Components
│   │       ├── __init__.py
│   │       ├── optical_flow_config.py     # ⚙️ Optical Flow Setup
│   │       ├── motion_detector.py         # 🔍 Motion Detection
│   │       ├── position_adjuster.py       # 📐 Position Adjustment
│   │       └── motion_visualizer.py      # 🎨 Motion Visualization
│   ├── team_assigner/                     # 🎨 Team Color Assignment
│   │   ├── __init__.py
│   │   └── team_assigner.py              # 🎨 Team Assignment Logic
│   ├── speed_and_distance_estimator/     # 📊 Speed & Distance Analysis
│   │   ├── __init__.py
│   │   └── speed_and_distance_estimator.py # 📊 Speed Analysis
│   ├── player_ball_assigner/              # ⚽ Ball Control Assignment
│   │   ├── __init__.py
│   │   └── player_ball_assigner.py       # ⚽ Ball Assignment Logic
│   ├── view_transformer/                  # 🔄 View Transformation
│   │   ├── __init__.py
│   │   └── view_transformer.py           # 🔄 View Transformation Logic
│   ├── utils/                             # 🛠️ Utility Functions
│   │   ├── __init__.py
│   │   ├── bbox_utils.py                 # 📦 Bounding Box Utilities
│   │   ├── parameter_scaler.py           # 📏 Parameter Scaling
│   │   └── video_utils.py                # 📹 Video Utilities
│   ├── models/                            # 🤖 AI Models
│   │   └── best.pt                       # 🎯 YOLO Model File
│   ├── input_videos/                      # 📹 Input Video Directory
│   │   └── CV_Task.mkv                   # 📹 Sample Input Video
│   ├── output_videos/                     # 📹 Output Video Directory
│   │   └── output_video.avi              # 📹 Processed Output Video
│   ├── stubs/                             # 💾 Cache Files
│   │   ├── camera_movement_stub.pkl      # 💾 Camera Movement Cache
│   │   └── track_stubs.pkl              # 💾 Tracking Cache
│   └── assets/                            # 🎨 Assets Directory
│       └── images/                       # 🖼️ Image Assets
├── docs/                                  # 📚 Documentation
│   ├── API_DOCS.md                       # 🔧 Technical Documentation
│   ├── ARCHITECTURE.md                   # 🏗️ System Architecture
│   └── TROUBLESHOOTING.md                # 🔧 Troubleshooting Guide
├── requirements.txt                       # 📦 Dependencies
└── README.md                             # 📖 Project Overview
```

---

## 🎯 Main Pipeline

### `src/main.py` - Main Pipeline Orchestrator

**Purpose**: Central coordination point for the entire football analysis pipeline.

**Responsibilities**:
- ✅ **Pipeline Orchestration**: Coordinates all processing stages
- ✅ **Module Coordination**: Manages interactions between modules
- ✅ **Error Handling**: Top-level error handling and recovery
- ✅ **Configuration Management**: Passes configuration to all modules

**Key Functions**:

#### `main()`
```python
def main():
    """
    Main pipeline orchestrator.
    
    Pipeline Flow:
    1. Configuration Setup → 2. Module Initialization → 3. Video I/O Setup
    4. Team Initialization → 5. Frame Processing → 6. Output Generation
    
    Returns:
        None
        
    Side Effects:
        - Processes input video
        - Generates output video with annotations
        - Creates cache files for faster subsequent runs
    """
    # 1. Setup configuration with resolution-based scaling
    config = setup_configuration()
    
    # 2. Initialize all modules with dependency injection
    modules = initialize_modules(config)
    
    # 3. Setup video I/O (input reader, output writer)
    video_gen, first_frame, out_writer = setup_video_io(config['input_path'], config['output_path'])
    
    # 4. Initialize team colors using K-Means clustering
    initialize_team_colors(modules['tracker'], modules['team_assigner'], modules['camera_estimator'], first_frame, config)
    
    # 5. Process video frames with comprehensive error handling
    process_video_frames(video_gen, first_frame, modules, config, out_writer)
```

**Dependencies**:
- `pipeline.config_manager`: Configuration setup
- `pipeline.module_initializer`: Module initialization
- `pipeline.video_setup`: Video I/O management
- `pipeline.team_initializer`: Team color detection
- `pipeline.frame_processor`: Frame processing loop

**Error Handling**:
- Comprehensive logging at all stages
- Graceful degradation on module failures
- Memory management and cleanup

---

## 🔄 Pipeline Layer

The pipeline layer contains high-level processing stages that coordinate the entire analysis workflow.

### `src/pipeline/` Directory

**Purpose**: Contains the main processing pipeline components that orchestrate the analysis workflow.

**Components**:
- `config_manager.py`: Configuration management and resolution-based scaling
- `module_initializer.py`: Module initialization and dependency injection
- `video_setup.py`: Video I/O management and setup
- `team_initializer.py`: Team color detection and initialization
- `frame_processor.py`: Main frame processing loop

---

### `src/pipeline/config_manager.py` - Configuration Management

**Purpose**: Centralized configuration management with resolution-based parameter scaling.

**Responsibilities**:
- ✅ **Configuration Setup**: Load and validate all configuration parameters
- ✅ **Resolution Detection**: Automatically detect video resolution
- ✅ **Parameter Scaling**: Scale parameters based on video resolution
- ✅ **Path Validation**: Validate input/output paths and model files

**Key Functions**:

#### `setup_configuration()`
```python
def setup_configuration():
    """
    Setup configuration with resolution-based parameter scaling.
    
    Returns:
        dict: Configuration dictionary with scaled parameters
        
    Configuration Keys:
        - MIN_PLAYERS_FOR_KMEANS (int): Minimum players for clustering
        - BALL_MAX_GAP_FRAMES (int): Max frames to carry forward lost ball
        - TEAM_CONFIDENCE_THRESHOLD (float): Min confidence for team assignment
        - TEAM_HYSTERESIS_FRAMES (int): Frames to confirm team switch
        - TEAM_MIN_COLOR_SEPARATION (float): Scaled based on video resolution
        - input_path (str): Path to input video
        - output_path (str): Path to output video
        - model_path (str): Path to YOLO model
        - track_stub_path (str): Path to tracking cache
        - camera_stub_path (str): Path to camera movement cache
        
    Example:
        >>> config = setup_configuration()
        >>> print(config['TEAM_MIN_COLOR_SEPARATION'])
        50.0  # Scaled based on video resolution
    """
```

#### `get_video_resolution(video_path)`
```python
def get_video_resolution(video_path):
    """
    Get video resolution safely.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Tuple[int, int] or None: (width, height) or None if detection fails
        
    Example:
        >>> resolution = get_video_resolution("video.mp4")
        >>> print(resolution)
        (1920, 1080)
    """
```

**Dependencies**:
- `cv2`: Video resolution detection
- `utils.parameter_scaler`: Parameter scaling functions
- `os`: Path validation

---

### `src/pipeline/module_initializer.py` - Module Initialization

**Purpose**: Initialize all processing modules with proper dependency injection.

**Responsibilities**:
- ✅ **Module Creation**: Create instances of all processing modules
- ✅ **Dependency Injection**: Pass dependencies between modules
- ✅ **Configuration Passing**: Pass configuration to all modules
- ✅ **Error Handling**: Handle module initialization errors

**Key Functions**:

#### `initialize_modules(config)`
```python
def initialize_modules(config):
    """
    Initialize all processing modules.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary containing all initialized modules
        
    Modules:
        - tracker: Object detection and tracking
        - camera_estimator: Camera motion estimation
        - team_assigner: Team color assignment
        - player_assigner: Ball control assignment
        - speed_distance_estimator: Speed and distance analysis
        - view_transformer: View transformation
        
    Example:
        >>> modules = initialize_modules(config)
        >>> print(modules.keys())
        dict_keys(['tracker', 'camera_estimator', 'team_assigner', ...])
    """
```

**Dependencies**:
- `trackers.tracker`: Main tracker module
- `camera_movement_estimator.camera_movement_estimator`: Camera motion module
- `team_assigner.team_assigner`: Team assignment module
- `player_ball_assigner.player_ball_assigner`: Ball assignment module
- `speed_and_distance_estimator.speed_and_distance_estimator`: Speed analysis module
- `view_transformer.view_transformer`: View transformation module

---

### `src/pipeline/video_setup.py` - Video I/O Management

**Purpose**: Handle video input/output operations and setup.

**Responsibilities**:
- ✅ **Video Reading**: Read input video files
- ✅ **Video Writing**: Setup output video writer
- ✅ **Format Support**: Support multiple video formats
- ✅ **Error Handling**: Handle video I/O errors

**Key Functions**:

#### `setup_video_io(input_path, output_path)`
```python
def setup_video_io(input_path, output_path):
    """
    Setup video input/output operations.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        
    Returns:
        tuple: (video_generator, first_frame, output_writer)
            - video_generator: Generator for reading frames
            - first_frame: First frame of the video
            - output_writer: VideoWriter for output
            
    Example:
        >>> video_gen, first_frame, writer = setup_video_io("input.mp4", "output.avi")
        >>> for frame in video_gen:
        ...     process_frame(frame)
    """
```

**Dependencies**:
- `cv2`: Video I/O operations
- `os`: Path validation

---

### `src/pipeline/team_initializer.py` - Team Color Detection

**Purpose**: Initialize team colors using K-Means clustering on the first frame.

**Responsibilities**:
- ✅ **Team Detection**: Detect team colors from first frame
- ✅ **K-Means Clustering**: Perform one-time team clustering
- ✅ **Color Validation**: Validate detected team colors
- ✅ **Configuration**: Pass team configuration to assigner

**Key Functions**:

#### `initialize_team_colors(tracker, team_assigner, camera_estimator, first_frame, config)`
```python
def initialize_team_colors(tracker, team_assigner, camera_estimator, first_frame, config):
    """
    Initialize team colors using K-Means clustering.
    
    Args:
        tracker: Tracker instance for object detection
        team_assigner: TeamAssigner instance for team assignment
        camera_estimator: CameraMovementEstimator instance
        first_frame (np.ndarray): First frame of the video
        config (dict): Configuration dictionary
        
    Process:
        1. Detect players in first frame
        2. Extract player colors
        3. Perform K-Means clustering
        4. Initialize team assigner with centroids
        
    Example:
        >>> initialize_team_colors(tracker, team_assigner, camera_estimator, first_frame, config)
        >>> print(team_assigner.team_colors)
        {1: [255, 0, 0], 2: [0, 0, 255]}
    """
```

**Dependencies**:
- `trackers.tracker`: Object detection
- `team_assigner.team_assigner`: Team assignment
- `camera_movement_estimator.camera_movement_estimator`: Camera motion

---

### `src/pipeline/frame_processor.py` - Frame Processing Loop

**Purpose**: Main frame processing loop that coordinates all analysis modules.

**Responsibilities**:
- ✅ **Frame Processing**: Process each video frame
- ✅ **Module Coordination**: Coordinate all processing modules
- ✅ **Error Handling**: Handle frame-level errors
- ✅ **Output Generation**: Generate annotated output frames

**Key Functions**:

#### `process_video_frames(video_generator, first_frame, modules, config, output_writer)`
```python
def process_video_frames(video_generator, first_frame, modules, config, output_writer):
    """
    Process video frames through the analysis pipeline.
    
    Args:
        video_generator: Generator for reading video frames
        first_frame (np.ndarray): First frame of the video
        modules (dict): Dictionary of processing modules
        config (dict): Configuration dictionary
        output_writer: VideoWriter for output
        
    Processing Pipeline:
        1. Object Detection & Tracking
        2. Camera Motion Estimation
        3. Team Assignment
        4. Ball Control Assignment
        5. Speed & Distance Analysis
        6. View Transformation
        7. Visualization & Output
        
    Example:
        >>> process_video_frames(video_gen, first_frame, modules, config, writer)
        # Processes all frames and generates output video
    """
```

**Dependencies**:
- All processing modules (tracker, camera_estimator, team_assigner, etc.)
- `cv2`: Video processing
- `logging`: Error logging

---

## 🎯 Core Modules

The core modules contain the main processing logic for each analysis component.

### `src/trackers/` Directory - Object Detection & Tracking

**Purpose**: Handles object detection, tracking, and visualization for players, referees, and ball.

**Components**:
- `tracker.py`: Main tracker orchestrator
- `components/`: Component-based architecture for tracker functionality

---

### `src/trackers/tracker.py` - Main Tracker Orchestrator

**Purpose**: Central orchestrator for object detection and tracking operations.

**Responsibilities**:
- ✅ **Component Coordination**: Coordinates all tracker components
- ✅ **Interface Management**: Provides clean interface to other modules
- ✅ **Error Handling**: Handles tracking errors gracefully
- ✅ **Performance Optimization**: Optimizes tracking performance

**Key Functions**:

#### `Tracker` Class
```python
class Tracker:
    """Main tracker orchestrator with component-based architecture."""
    
    def __init__(self, model_path):
        """
        Initialize tracker with model path.
        
        Args:
            model_path (str): Path to YOLO model file
            
        Components:
            - model_manager: YOLO model management
            - detection_processor: Detection processing
            - tracking_engine: ByteTrack engine
            - visualizer: Visualization
        """
        self.model_manager = ModelManager(model_path)
        self.detection_processor = DetectionProcessor(self.model_manager.model)
        self.tracking_engine = TrackingEngine(self.model_manager.model, self.model_manager.tracker)
        self.visualizer = Visualizer()
```

#### `get_object_tracks(frames, read_from_stub=False, stub_path=None)`
```python
def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    """
    Get object tracks from video frames.
    
    Args:
        frames (List[np.ndarray]): List of video frames
        read_from_stub (bool): Whether to read from cache
        stub_path (str): Path to cache file
        
    Returns:
        dict: Tracking results with 'players', 'referees', 'ball' keys
        
    Example:
        >>> tracker = Tracker("models/best.pt")
        >>> tracks = tracker.get_object_tracks(frames)
        >>> print(tracks['players'][0])
        {1: {'bbox': [100, 200, 150, 250], 'confidence': 0.95}}
    """
```

**Dependencies**:
- `trackers.components.model_manager`: Model management
- `trackers.components.detection_processor`: Detection processing
- `trackers.components.tracking_engine`: Tracking engine
- `trackers.components.visualizer`: Visualization

---

### `src/trackers/components/` Directory - Tracker Components

**Purpose**: Component-based architecture for tracker functionality.

**Components**:
- `model_manager.py`: YOLO model management with fallback
- `detection_processor.py`: Detection processing and memory management
- `tracking_engine.py`: ByteTrack tracking engine
- `visualizer.py`: Drawing and visualization
- `track_utils.py`: Utility functions for tracking

---

### `src/trackers/components/model_manager.py` - YOLOv5 Model Management

**Purpose**: Manages YOLOv5 model loading with fallback support.

**Responsibilities**:
- ✅ **Model Loading**: Load YOLOv5 model with error handling
- ✅ **Fallback Support**: Use fallback model if custom model fails
- ✅ **ByteTrack Setup**: Initialize ByteTrack tracker
- ✅ **Error Handling**: Handle model loading errors gracefully

**Key Functions**:

#### `ModelManager` Class
```python
class ModelManager:
    """YOLOv5 model management with fallback support."""
    
    def __init__(self, model_path):
        """
        Initialize model manager with fallback support.
        
        Args:
            model_path (str): Path to YOLOv5 model
            
        Fallback Strategy:
            1. Try to load custom model
            2. If fails, load fallback model (yolov5su.pt)
            3. If both fail, raise exception
            
        Raises:
            Exception: If both custom and fallback models fail to load
        """
        try:
            self.model = YOLO(model_path)
            self.tracker = sv.ByteTrack()
            print(f"✅ Successfully loaded custom model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            print("🔄 Trying to use a smaller model as fallback...")
            self.model = YOLO('yolov5su.pt')
            self.tracker = sv.ByteTrack()
            print("✅ Successfully loaded fallback model: yolov5su.pt")
```

**Dependencies**:
- `ultralytics`: YOLOv5 model loading
- `supervision`: ByteTrack tracker

---

### `src/trackers/components/detection_processor.py` - Detection Processing

**Purpose**: Handles object detection and memory management.

**Responsibilities**:
- ✅ **Frame Detection**: Detect objects in video frames
- ✅ **Memory Management**: Efficient memory usage
- ✅ **Error Handling**: Handle detection errors
- ✅ **Performance Optimization**: Optimize detection performance

**Key Functions**:

#### `detect_frames(frames)`
```python
def detect_frames(self, frames):
    """
    Detect objects in video frames.
    
    Args:
        frames (List[np.ndarray]): List of video frames
        
    Returns:
        List[dict]: Detection results for each frame
        
    Detection Classes:
        - 0: Person (players, referees)
        - 32: Sports ball
        
    Example:
        >>> processor = DetectionProcessor(model)
        >>> detections = processor.detect_frames(frames)
        >>> print(detections[0])
        [{'bbox': [100, 200, 150, 250], 'confidence': 0.95, 'class': 0}]
    """
```

**Dependencies**:
- `ultralytics`: YOLOv5 model
- `numpy`: Array operations
- `gc`: Garbage collection

---

### `src/trackers/components/tracking_engine.py` - ByteTrack Engine

**Purpose**: Implements ByteTrack algorithm for object tracking.

**Responsibilities**:
- ✅ **Object Tracking**: Track objects across frames
- ✅ **ID Management**: Consistent ID assignment
- ✅ **Data Structure Management**: Manage tracking data structures
- ✅ **Performance Optimization**: Optimize tracking performance

**Key Functions**:

#### `get_object_tracks(frames, read_from_stub=False, stub_path=None)`
```python
def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    """
    Get object tracks using ByteTrack algorithm.
    
    Args:
        frames (List[np.ndarray]): List of video frames
        read_from_stub (bool): Whether to read from cache
        stub_path (str): Path to cache file
        
    Returns:
        dict: Tracking results with 'players', 'referees', 'ball' keys
        
    Algorithm:
        1. Detect objects in each frame
        2. Track objects using ByteTrack
        3. Classify objects (players, referees, ball)
        4. Return tracking results
        
    Example:
        >>> engine = TrackingEngine(model, tracker)
        >>> tracks = engine.get_object_tracks(frames)
        >>> print(tracks['players'][0])
        {1: {'bbox': [100, 200, 150, 250], 'confidence': 0.95}}
    """
```

**Dependencies**:
- `supervision`: ByteTrack tracker
- `ultralytics`: YOLOv5 model
- `numpy`: Array operations

---

### `src/trackers/components/visualizer.py` - Visualization

**Purpose**: Handles drawing and visualization of tracking results.

**Responsibilities**:
- ✅ **Drawing**: Draw bounding boxes and annotations
- ✅ **Visualization**: Create visual representations
- ✅ **Customization**: Configurable visualization options
- ✅ **Performance**: Efficient rendering

**Key Functions**:

#### `draw_annotations(frame, tracks, frame_idx)`
```python
def draw_annotations(self, frame, tracks, frame_idx):
    """
    Draw annotations on frame.
    
    Args:
        frame (np.ndarray): Video frame
        tracks (dict): Tracking results
        frame_idx (int): Frame index
        
    Returns:
        np.ndarray: Annotated frame
        
    Annotations:
        - Bounding boxes for players, referees, ball
        - Team colors
        - Speed information
        - ID labels
        
    Example:
        >>> visualizer = Visualizer()
        >>> annotated_frame = visualizer.draw_annotations(frame, tracks, 0)
        >>> cv2.imshow("Annotated", annotated_frame)
    """
```

**Dependencies**:
- `cv2`: Drawing operations
- `numpy`: Array operations

---

### `src/trackers/components/track_utils.py` - Utility Functions

**Purpose**: Utility functions for tracking operations.

**Responsibilities**:
- ✅ **Position Calculation**: Calculate object positions
- ✅ **Ball Interpolation**: Interpolate ball positions
- ✅ **Data Processing**: Process tracking data
- ✅ **Helper Functions**: Utility functions for tracking

**Key Functions**:

#### `add_position_to_tracks(tracks)`
```python
def add_position_to_tracks(tracks):
    """
    Add position information to tracks.
    
    Args:
        tracks (dict): Tracking results
        
    Returns:
        dict: Updated tracks with position information
        
    Position Information:
        - center_x: Center X coordinate
        - center_y: Center Y coordinate
        - width: Bounding box width
        - height: Bounding box height
        
    Example:
        >>> updated_tracks = add_position_to_tracks(tracks)
        >>> print(updated_tracks['players'][0][1]['center_x'])
        125.0
    """
```

#### `interpolate_ball_positions(tracks)`
```python
def interpolate_ball_positions(tracks):
    """
    Interpolate ball positions for missing frames.
    
    Args:
        tracks (dict): Tracking results
        
    Returns:
        dict: Updated tracks with interpolated ball positions
        
    Interpolation:
        - Linear interpolation between known positions
        - Handles missing ball frames
        - Maintains smooth ball trajectory
        
    Example:
        >>> updated_tracks = interpolate_ball_positions(tracks)
        >>> print(updated_tracks['ball'][5])
        {'bbox': [100, 200, 110, 210], 'interpolated': True}
    """
```

**Dependencies**:
- `numpy`: Array operations
- `scipy`: Interpolation functions

---

## 📹 Camera Motion Estimation

### `src/camera_movement_estimator/` Directory - Camera Motion Compensation

**Purpose**: Handles camera motion detection and position compensation.

**Components**:
- `camera_movement_estimator.py`: Main estimator orchestrator
- `components/`: Component-based architecture for motion detection

---

### `src/camera_movement_estimator/camera_movement_estimator.py` - Main Estimator

**Purpose**: Central orchestrator for camera motion estimation and compensation.

**Responsibilities**:
- ✅ **Motion Detection**: Detect camera movement across frames
- ✅ **Position Compensation**: Adjust object positions for camera motion
- ✅ **Component Coordination**: Coordinate motion detection components
- ✅ **Error Handling**: Handle motion detection errors gracefully

**Key Functions**:

#### `CameraMovementEstimator` Class
```python
class CameraMovementEstimator:
    """Camera movement estimation with median consensus."""
    
    def __init__(self, config):
        """
        Initialize camera movement estimator.
        
        Args:
            config (dict): Configuration dictionary
            
        Components:
            - optical_flow_config: Optical flow configuration
            - motion_detector: Motion detection engine
            - position_adjuster: Position adjustment logic
            - motion_visualizer: Motion visualization
        """
        self.config = OpticalFlowConfig(config)
        self.motion_detector = MotionDetector(self.config)
        self.position_adjuster = PositionAdjuster()
        self.visualizer = MotionVisualizer()
```

#### `detect_camera_movement(frames, read_from_stub=False, stub_path=None)`
```python
def detect_camera_movement(self, frames, read_from_stub=False, stub_path=None):
    """
    Detect camera movement using median consensus.
    
    Args:
        frames (List[np.ndarray]): List of video frames
        read_from_stub (bool): Whether to read from cache
        stub_path (str): Path to cache file
        
    Returns:
        List[List[float]]: Camera movement vectors for each frame
        
    Algorithm:
        1. Extract features using goodFeaturesToTrack
        2. Track features using calcOpticalFlowPyrLK
        3. Calculate motion vectors for valid features
        4. Apply median consensus to eliminate outliers
        
    Example:
        >>> estimator = CameraMovementEstimator(config)
        >>> movements = estimator.detect_camera_movement(frames)
        >>> print(movements[0])
        [0.5, -0.2]  # [x_movement, y_movement]
    """
```

**Dependencies**:
- `camera_movement_estimator.components.optical_flow_config`: Optical flow setup
- `camera_movement_estimator.components.motion_detector`: Motion detection
- `camera_movement_estimator.components.position_adjuster`: Position adjustment
- `camera_movement_estimator.components.motion_visualizer`: Motion visualization

---

### `src/camera_movement_estimator/components/` Directory - Motion Components

**Purpose**: Component-based architecture for camera motion detection.

**Components**:
- `optical_flow_config.py`: Optical flow configuration and setup
- `motion_detector.py`: Motion detection using Lucas-Kanade optical flow
- `position_adjuster.py`: Position adjustment for camera motion
- `motion_visualizer.py`: Motion visualization and debugging

---

### `src/camera_movement_estimator/components/motion_detector.py` - Motion Detection

**Purpose**: Implements camera motion detection using Lucas-Kanade optical flow with median consensus.

**Responsibilities**:
- ✅ **Feature Extraction**: Extract features using goodFeaturesToTrack
- ✅ **Optical Flow**: Track features using calcOpticalFlowPyrLK
- ✅ **Motion Calculation**: Calculate motion vectors for valid features
- ✅ **Median Consensus**: Apply median consensus to eliminate outliers

**Key Functions**:

#### `detect_camera_movement(frames, read_from_stub=False, stub_path=None)`
```python
def detect_camera_movement(self, frames, read_from_stub=False, stub_path=None):
    """
    Detect camera movement using Lucas-Kanade optical flow with median consensus.
    
    Args:
        frames (List[np.ndarray]): Video frames
        read_from_stub (bool): Read from cache if available
        stub_path (str): Path to cache file
        
    Returns:
        List[List[float]]: Camera movement vectors
        
    Algorithm:
        1. Extract features using goodFeaturesToTrack
        2. Track features using calcOpticalFlowPyrLK
        3. Calculate motion vectors for valid features
        4. Apply median consensus to eliminate outliers
        
    Example:
        >>> detector = MotionDetector(config)
        >>> movements = detector.detect_camera_movement(frames)
        >>> print(len(movements))
        100  # One movement vector per frame
    """
```

**Dependencies**:
- `cv2`: OpenCV for optical flow
- `numpy`: Array operations
- `pickle`: Cache management

---

### `src/camera_movement_estimator/components/position_adjuster.py` - Position Adjustment

**Purpose**: Adjusts object positions based on camera motion.

**Responsibilities**:
- ✅ **Position Compensation**: Compensate for camera motion
- ✅ **Coordinate Transformation**: Transform coordinates based on motion
- ✅ **Error Handling**: Handle position adjustment errors
- ✅ **Performance Optimization**: Optimize position adjustment

**Key Functions**:

#### `add_adjust_positions_to_tracks(tracks, camera_movements)`
```python
def add_adjust_positions_to_tracks(self, tracks, camera_movements):
    """
    Adjust object positions based on camera motion.
    
    Args:
        tracks (dict): Tracking results
        camera_movements (List[List[float]]): Camera movement vectors
        
    Returns:
        dict: Updated tracks with adjusted positions
        
    Adjustment:
        - Compensate for camera motion
        - Maintain relative positions
        - Handle missing movements
        
    Example:
        >>> adjuster = PositionAdjuster()
        >>> adjusted_tracks = adjuster.add_adjust_positions_to_tracks(tracks, movements)
        >>> print(adjusted_tracks['players'][0][1]['adjusted_bbox'])
        [95, 195, 145, 245]  # Adjusted for camera motion
    """
```

**Dependencies**:
- `numpy`: Array operations
- `utils.bbox_utils`: Bounding box utilities

---

### `src/camera_movement_estimator/components/motion_visualizer.py` - Motion Visualization

**Purpose**: Visualizes camera motion for debugging and analysis.

**Responsibilities**:
- ✅ **Motion Visualization**: Draw motion vectors
- ✅ **Debug Information**: Display debug information
- ✅ **Performance Monitoring**: Monitor motion detection performance
- ✅ **Error Visualization**: Visualize motion detection errors

**Key Functions**:

#### `draw_camera_movement(frame, movements, frame_idx)`
```python
def draw_camera_movement(self, frame, movements, frame_idx):
    """
    Draw camera movement visualization.
    
    Args:
        frame (np.ndarray): Video frame
        movements (List[float]): Camera movement vector
        frame_idx (int): Frame index
        
    Returns:
        np.ndarray: Frame with motion visualization
        
    Visualization:
        - Motion vectors
        - Motion magnitude
        - Debug information
        
    Example:
        >>> visualizer = MotionVisualizer()
        >>> vis_frame = visualizer.draw_camera_movement(frame, [0.5, -0.2], 0)
        >>> cv2.imshow("Motion", vis_frame)
    """
```

**Dependencies**:
- `cv2`: Drawing operations
- `numpy`: Array operations

---

## 🎨 Team Assignment

### `src/team_assigner/` Directory - Team Color Assignment

**Purpose**: Handles team color assignment using K-Means clustering.

**Components**:
- `team_assigner.py`: Team assignment logic with K-Means optimization

---

### `src/team_assigner/team_assigner.py` - Team Assignment Logic

**Purpose**: Assigns team colors to players using optimized K-Means clustering.

**Responsibilities**:
- ✅ **Team Detection**: Detect team colors using K-Means clustering
- ✅ **Color Classification**: Classify players based on team colors
- ✅ **Performance Optimization**: One-time clustering with cached centroids
- ✅ **Error Handling**: Handle team assignment errors

**Key Functions**:

#### `TeamAssigner` Class
```python
class TeamAssigner:
    """Team assignment with K-Means optimization."""
    
    def __init__(self, config=None):
        """
        Initialize team assigner with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary
            
        Configuration:
            - TEAM_MIN_COLOR_SEPARATION: Minimum color separation
            - TEAM_CONFIDENCE_THRESHOLD: Confidence threshold
            - TEAM_HYSTERESIS_FRAMES: Hysteresis frames
        """
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}
        
        if config and 'TEAM_MIN_COLOR_SEPARATION_SCALED' in config:
            self.min_color_separation = config['TEAM_MIN_COLOR_SEPARATION_SCALED']
        else:
            self.min_color_separation = 50.0
            
        self.confidence_threshold = config.get('TEAM_CONFIDENCE_THRESHOLD', 0.6) if config else 0.6
        self.hysteresis_frames = config.get('TEAM_HYSTERESIS_FRAMES', 5) if config else 5
```

#### `get_player_team(frame, bbox)`
```python
def get_player_team(self, frame, bbox):
    """
    Get team assignment for a player using optimized K-Means.
    
    Args:
        frame (np.ndarray): Video frame
        bbox (List[float]): Bounding box [x1, y1, x2, y2]
        
    Returns:
        int: Team ID (1 or 2) or -1 if unknown
        
    Algorithm:
        1. Extract player color from bounding box
        2. Use cached K-Means centroids for classification
        3. Apply confidence threshold and hysteresis
        
    Example:
        >>> assigner = TeamAssigner(config)
        >>> team_id = assigner.get_player_team(frame, [100, 200, 150, 250])
        >>> print(team_id)
        1  # Team 1
    """
```

#### `initialize_team_colors(frame, player_bboxes)`
```python
def initialize_team_colors(self, frame, player_bboxes):
    """
    Initialize team colors using K-Means clustering.
    
    Args:
        frame (np.ndarray): Video frame
        player_bboxes (List[List[float]]): List of player bounding boxes
        
    Algorithm:
        1. Extract colors from all player bounding boxes
        2. Perform K-Means clustering with 2 clusters
        3. Cache centroids for subsequent frames
        
    Example:
        >>> assigner = TeamAssigner()
        >>> assigner.initialize_team_colors(frame, player_bboxes)
        >>> print(assigner.team_colors)
        {1: [255, 0, 0], 2: [0, 0, 255]}  # Team colors
    """
```

**Dependencies**:
- `sklearn.cluster.KMeans`: K-Means clustering
- `numpy`: Array operations
- `cv2`: Image processing

---

## 📊 Speed & Distance Analysis

### `src/speed_and_distance_estimator/` Directory - Speed & Distance Analysis

**Purpose**: Handles speed and distance calculations with exponential smoothing.

**Components**:
- `speed_and_distance_estimator.py`: Speed and distance analysis with smoothing

---

### `src/speed_and_distance_estimator/speed_and_distance_estimator.py` - Speed Analysis

**Purpose**: Calculates speed and distance with exponential smoothing for smooth output.

**Responsibilities**:
- ✅ **Speed Calculation**: Calculate speed from position changes
- ✅ **Distance Tracking**: Track cumulative distance
- ✅ **Exponential Smoothing**: Apply smoothing for continuous output
- ✅ **Performance Optimization**: Optimize speed calculations

**Key Functions**:

#### `SpeedAndDistanceEstimator` Class
```python
class SpeedAndDistanceEstimator:
    """Speed and distance estimation with exponential smoothing."""
    
    def __init__(self):
        """Initialize speed estimator with smoothing infrastructure."""
        self.frame_window = 5
        self.frame_rate = 24
        self.smoothed_speeds = {}  # track_id -> last_smoothed_speed
        self.smoothing_alpha = 0.3  # Configurable smoothing parameter
```

#### `smooth_speed_exponential(track_id, new_speed)`
```python
def smooth_speed_exponential(self, track_id, new_speed):
    """
    Apply exponential smoothing to speed measurements.
    
    Formula: smoothed = α * new_speed + (1-α) * previous_smoothed
    
    Args:
        track_id (str): Unique identifier for the track
        new_speed (float): Raw speed value from calculation (km/h)
        
    Returns:
        float: Smoothed speed value (km/h)
        
    Example:
        >>> estimator = SpeedAndDistanceEstimator()
        >>> smoothed = estimator.smooth_speed_exponential("player_1", 15.5)
        >>> print(smoothed)
        14.2  # Smoothed speed
    """
```

#### `add_speed_and_distance_to_tracks(tracks)`
```python
def add_speed_and_distance_to_tracks(self, tracks):
    """
    Add speed and distance calculations to tracks with smoothing.
    
    Args:
        tracks (dict): Tracking results dictionary
        
    Returns:
        dict: Updated tracks with speed and distance data
        
    Calculations:
        - Speed: Distance / time (km/h)
        - Distance: Cumulative distance traveled
        - Smoothing: Exponential smoothing for continuous output
        
    Example:
        >>> estimator = SpeedAndDistanceEstimator()
        >>> updated_tracks = estimator.add_speed_and_distance_to_tracks(tracks)
        >>> print(updated_tracks['players'][0][1]['speed'])
        12.5  # Speed in km/h
    """
```

**Dependencies**:
- `numpy`: Array operations
- `math`: Mathematical functions

---

## ⚽ Ball Control Assignment

### `src/player_ball_assigner/` Directory - Ball Control Assignment

**Purpose**: Assigns ball control to players based on proximity and team assignment.

**Components**:
- `player_ball_assigner.py`: Ball control assignment logic

---

### `src/player_ball_assigner/player_ball_assigner.py` - Ball Assignment Logic

**Purpose**: Determines which player has ball control based on proximity and team assignment.

**Responsibilities**:
- ✅ **Ball Control Detection**: Detect which player has ball control
- ✅ **Proximity Analysis**: Analyze ball-player proximity
- ✅ **Team Assignment**: Assign ball control to team
- ✅ **Error Handling**: Handle missing ball or player data

**Key Functions**:

#### `PlayerBallAssigner` Class
```python
class PlayerBallAssigner:
    """Ball control assignment based on proximity and team assignment."""
    
    def __init__(self, config=None):
        """
        Initialize ball assigner with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.ball_control_threshold = self.config.get('BALL_CONTROL_THRESHOLD', 50.0)
```

#### `assign_ball_control(players, ball, team_assignments)`
```python
def assign_ball_control(self, players, ball, team_assignments):
    """
    Assign ball control to players based on proximity and team.
    
    Args:
        players (dict): Player tracking data
        ball (dict): Ball tracking data
        team_assignments (dict): Team assignments for players
        
    Returns:
        dict: Ball control assignments
        
    Algorithm:
        1. Calculate distance between ball and each player
        2. Find closest player to ball
        3. Assign ball control to player's team
        4. Handle missing ball or player data
        
    Example:
        >>> assigner = PlayerBallAssigner()
        >>> control = assigner.assign_ball_control(players, ball, teams)
        >>> print(control)
        {'team': 1, 'player_id': 5, 'distance': 25.3}
    """
```

**Dependencies**:
- `numpy`: Array operations
- `math`: Mathematical functions

---

## 🔄 View Transformation

### `src/view_transformer/` Directory - View Transformation

**Purpose**: Handles view transformation and coordinate system conversions.

**Components**:
- `view_transformer.py`: View transformation logic

---

### `src/view_transformer/view_transformer.py` - View Transformation Logic

**Purpose**: Transforms coordinates between different view systems.

**Responsibilities**:
- ✅ **Coordinate Transformation**: Transform coordinates between views
- ✅ **View Conversion**: Convert between different coordinate systems
- ✅ **Error Handling**: Handle transformation errors
- ✅ **Performance Optimization**: Optimize transformation operations

**Key Functions**:

#### `ViewTransformer` Class
```python
class ViewTransformer:
    """View transformation and coordinate system conversions."""
    
    def __init__(self, config=None):
        """
        Initialize view transformer with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.transformation_matrix = None
```

#### `transform_coordinates(coordinates, transformation_type)`
```python
def transform_coordinates(self, coordinates, transformation_type):
    """
    Transform coordinates between different view systems.
    
    Args:
        coordinates (List[float]): Input coordinates
        transformation_type (str): Type of transformation
        
    Returns:
        List[float]: Transformed coordinates
        
    Transformation Types:
        - 'pitch_to_screen': Pitch coordinates to screen coordinates
        - 'screen_to_pitch': Screen coordinates to pitch coordinates
        - 'normalize': Normalize coordinates to [0, 1] range
        
    Example:
        >>> transformer = ViewTransformer()
        >>> transformed = transformer.transform_coordinates([100, 200], 'pitch_to_screen')
        >>> print(transformed)
        [150, 250]  # Transformed coordinates
    """
```

**Dependencies**:
- `numpy`: Array operations
- `cv2`: Transformation operations

---

## 🛠️ Utility Functions

### `src/utils/` Directory - Utility Functions

**Purpose**: Contains utility functions used across the system.

**Components**:
- `bbox_utils.py`: Bounding box utilities
- `parameter_scaler.py`: Parameter scaling functions
- `video_utils.py`: Video utility functions

---

### `src/utils/bbox_utils.py` - Bounding Box Utilities

**Purpose**: Utility functions for bounding box operations.

**Responsibilities**:
- ✅ **Bounding Box Validation**: Validate bounding box format and bounds
- ✅ **Format Checking**: Check bounding box format
- ✅ **Error Handling**: Handle validation errors
- ✅ **Performance Optimization**: Optimize validation operations

**Key Functions**:

#### `is_valid_bbox(bbox, frame_shape=None)`
```python
def is_valid_bbox(bbox, frame_shape=None):
    """
    Validate bounding box format and bounds.
    
    Args:
        bbox (List[float]): Bounding box [x1, y1, x2, y2]
        frame_shape (Tuple[int, int], optional): Frame dimensions (height, width)
        
    Returns:
        bool: True if bbox is valid, False otherwise
        
    Validation Checks:
        - Format: Must be list/tuple with 4 elements
        - Bounds: x1 < x2, y1 < y2
        - Frame bounds: Within frame dimensions
        - NaN values: No NaN or infinite values
        
    Example:
        >>> is_valid_bbox([100, 200, 150, 250])
        True
        >>> is_valid_bbox([150, 200, 100, 250])  # Invalid: x1 > x2
        False
    """
```

**Dependencies**:
- `numpy`: Array operations
- `math`: Mathematical functions

---

### `src/utils/parameter_scaler.py` - Parameter Scaling

**Purpose**: Scales parameters based on video resolution.

**Responsibilities**:
- ✅ **Resolution Detection**: Detect video resolution
- ✅ **Parameter Scaling**: Scale parameters based on resolution
- ✅ **Error Handling**: Handle scaling errors
- ✅ **Performance Optimization**: Optimize scaling operations

**Key Functions**:

#### `scale_for_resolution(value, current_resolution, base_resolution=(1920, 1080))`
```python
def scale_for_resolution(value, current_resolution, base_resolution=(1920, 1080)):
    """
    Scale pixel-based parameters based on video resolution.
    
    Args:
        value (float): Base parameter value (calibrated at base_resolution)
        current_resolution (Tuple[int, int]): Current video resolution (width, height)
        base_resolution (Tuple[int, int]): Reference resolution (default 1920x1080)
    
    Returns:
        float: Scaled parameter value
        
    Examples:
        >>> # Scale for 720p video
        >>> scale_for_resolution(50.0, (1280, 720))
        33.333333333333336
        
        >>> # Scale for 4K video
        >>> scale_for_resolution(50.0, (3840, 2160))
        100.0
        
        >>> # Fallback if resolution is None
        >>> scale_for_resolution(50.0, None)
        50.0
    """
```

**Dependencies**:
- `numpy`: Array operations

---

### `src/utils/video_utils.py` - Video Utilities

**Purpose**: Utility functions for video processing.

**Responsibilities**:
- ✅ **Video Processing**: Video utility functions
- ✅ **Format Support**: Support multiple video formats
- ✅ **Error Handling**: Handle video processing errors
- ✅ **Performance Optimization**: Optimize video operations

**Key Functions**:

#### `get_video_info(video_path)`
```python
def get_video_info(video_path):
    """
    Get video information.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video information dictionary
        
    Information:
        - width: Video width
        - height: Video height
        - fps: Frames per second
        - frame_count: Total frame count
        - duration: Video duration in seconds
        
    Example:
        >>> info = get_video_info("video.mp4")
        >>> print(info)
        {'width': 1920, 'height': 1080, 'fps': 24.0, 'frame_count': 2400, 'duration': 100.0}
    """
```

**Dependencies**:
- `cv2`: Video processing
- `os`: Path operations

---

## ⚙️ Configuration System

### Configuration Management

The system uses a centralized configuration system with resolution-based parameter scaling.

**Configuration Keys**:
- `MIN_PLAYERS_FOR_KMEANS`: Minimum players for clustering
- `BALL_MAX_GAP_FRAMES`: Max frames to carry forward lost ball
- `TEAM_CONFIDENCE_THRESHOLD`: Min confidence for team assignment
- `TEAM_HYSTERESIS_FRAMES`: Frames to confirm team switch
- `TEAM_MIN_COLOR_SEPARATION`: Scaled based on video resolution
- `input_path`: Path to input video
- `output_path`: Path to output video
- `model_path`: Path to YOLO model
- `track_stub_path`: Path to tracking cache
- `camera_stub_path`: Path to camera movement cache

**Resolution-Based Scaling**:
```python
# Automatic parameter scaling based on video resolution
TEAM_MIN_COLOR_SEPARATION = scale_for_resolution(50.0, video_resolution)
```

---

## 🛡️ Error Handling

### Error Types and Handling

#### 1. **Missing Detections**
```python
# In tracking_engine.py
if detection is None:
    tracks["players"].append({})
    tracks["referees"].append({})
    tracks["ball"].append({})
    continue
```

#### 2. **Invalid Bounding Boxes**
```python
# In bbox_utils.py
def is_valid_bbox(bbox, frame_shape=None):
    if bbox is None or len(bbox) != 4:
        return False
    if any(math.isnan(x) or math.isinf(x) for x in bbox):
        return False
    # ... additional validation
```

#### 3. **Lost Ball Frames**
```python
# In frame_processor.py
if ball_missing_count <= config['BALL_MAX_GAP_FRAMES'] and team_ball_control:
    team_ball_control.append(team_ball_control[-1])  # carry forward
else:
    team_ball_control.append(-1)  # unknown team
```

#### 4. **Memory Errors**
```python
# In motion_detector.py
try:
    # Process frames
    pass
except MemoryError:
    gc.collect()  # Force garbage collection
    # Retry with smaller batch
```

### Error Logging

```python
import logging

logger = logging.getLogger("football_analysis")

# Log errors with context
logger.error(f"Failed to process frame {frame_num}: {error}")
logger.warning(f"Low confidence detection: {confidence}")
logger.info(f"Successfully processed {frame_count} frames")
```

---

## ⚡ Performance Optimizations

### Algorithm Optimizations

#### **K-Means Optimization**
```python
# Before: Re-clustering every frame (O(n) per frame)
# After: One-time clustering + distance comparison (O(1) per frame)
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=20, max_iter=300, random_state=42)
kmeans.fit(player_colors)
# Cache centroids for subsequent frames
predicted_team = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
```

#### **Median Consensus**
```python
# Robust motion estimation across multiple features
valid_motions = []
for i, (old_pt, new_pt, st) in enumerate(zip(old_features, new_features, status)):
    if st == 1:  # valid feature
        motion = [old_pt[0] - new_pt[0], old_pt[1] - new_pt[1]]
        valid_motions.append(motion)

if len(valid_motions) > 0:
    movements_array = np.array(valid_motions)
    camera_movement_x = float(np.median(movements_array[:, 0]) / self.config.scale)
    camera_movement_y = float(np.median(movements_array[:, 1]) / self.config.scale)
```

### Memory Management

#### **Automatic Memory Cleanup**
```python
import gc

# Force garbage collection after processing
gc.collect()

# Downscale frames for memory efficiency
old_gray = cv2.resize(old_gray_full, None, fx=scale, fy=scale)
```

#### **Efficient Data Structures**
```python
# Use efficient data structures
self.smoothed_speeds = {}  # Dictionary for O(1) lookup
self.team_colors = {}      # Cached centroids
```

### Caching Strategies

#### **Stub Files**
```python
# Cache processing results
if read_from_stub and os.path.exists(stub_path):
    with open(stub_path, 'rb') as f:
        return pickle.load(f)
```

#### **Centroid Caching**
```python
# Cache K-Means centroids
self.kmeans = kmeans
self.team_colors[1] = kmeans.cluster_centers_[0]
self.team_colors[2] = kmeans.cluster_centers_[1]
```

---

*This comprehensive technical documentation covers all major components, modules, and functions of the Football Analysis System. Each section provides detailed information about purpose, responsibilities, key functions, dependencies, and usage examples.*

---

## 🎨 Team Assignment

### `team_assigner/team_assigner.py`

#### `TeamAssigner` Class

```python
class TeamAssigner:
    """Team assignment with K-Means optimization."""
    
    def __init__(self, config=None):
        """
        Initialize team assigner with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}
        
        if config and 'TEAM_MIN_COLOR_SEPARATION_SCALED' in config:
            self.min_color_separation = config['TEAM_MIN_COLOR_SEPARATION_SCALED']
        else:
            self.min_color_separation = 50.0
            
        self.confidence_threshold = config.get('TEAM_CONFIDENCE_THRESHOLD', 0.6) if config else 0.6
        self.hysteresis_frames = config.get('TEAM_HYSTERESIS_FRAMES', 5) if config else 5
```

#### `get_player_team(frame, bbox)`

```python
def get_player_team(self, frame, bbox):
    """
    Get team assignment for a player using optimized K-Means.
    
    Algorithm:
    1. Extract player color from bounding box
    2. Use cached K-Means centroids for classification
    3. Apply confidence threshold and hysteresis
    
    Args:
        frame (np.ndarray): Video frame
        bbox (List[float]): Bounding box [x1, y1, x2, y2]
        
    Returns:
        int: Team ID (1 or 2) or -1 if unknown
        
    Example:
        >>> assigner = TeamAssigner(config)
        >>> team_id = assigner.get_player_team(frame, [100, 200, 150, 250])
        >>> print(team_id)
        1  # Team 1
    """
```

#### `initialize_team_colors(frame, player_bboxes)`

```python
def initialize_team_colors(self, frame, player_bboxes):
    """
    Initialize team colors using K-Means clustering.
    
    Algorithm:
    1. Extract colors from all player bounding boxes
    2. Perform K-Means clustering with 2 clusters
    3. Cache centroids for subsequent frames
    
    Args:
        frame (np.ndarray): Video frame
        player_bboxes (List[List[float]]): List of player bounding boxes
        
    Example:
        >>> assigner = TeamAssigner()
        >>> assigner.initialize_team_colors(frame, player_bboxes)
        >>> print(assigner.team_colors)
        {1: [255, 0, 0], 2: [0, 0, 255]}  # Team colors
    """
```

---

## 📊 Speed & Distance Analysis

### `speed_and_distance_estimator/speed_and_distance_estimator.py`

#### `SpeedAndDistanceEstimator` Class

```python
class SpeedAndDistanceEstimator:
    """Speed and distance estimation with exponential smoothing."""
    
    def __init__(self):
        """Initialize speed estimator with smoothing infrastructure."""
        self.frame_window = 5
        self.frame_rate = 24
        self.smoothed_speeds = {}  # track_id -> last_smoothed_speed
        self.smoothing_alpha = 0.3  # Configurable smoothing parameter
```

#### `smooth_speed_exponential(track_id, new_speed)`

```python
def smooth_speed_exponential(self, track_id, new_speed):
    """
    Apply exponential smoothing to speed measurements.
    
    Formula: smoothed = α * new_speed + (1-α) * previous_smoothed
    
    Args:
        track_id (str): Unique identifier for the track
        new_speed (float): Raw speed value from calculation (km/h)
        
    Returns:
        float: Smoothed speed value (km/h)
        
    Example:
        >>> estimator = SpeedAndDistanceEstimator()
        >>> smoothed = estimator.smooth_speed_exponential("player_1", 15.5)
        >>> print(smoothed)
        14.2  # Smoothed speed
    """
```

#### `add_speed_and_distance_to_tracks(tracks)`

```python
def add_speed_and_distance_to_tracks(self, tracks):
    """
    Add speed and distance calculations to tracks with smoothing.
    
    Args:
        tracks (dict): Tracking results dictionary
        
    Returns:
        dict: Updated tracks with speed and distance data
        
    Example:
        >>> estimator = SpeedAndDistanceEstimator()
        >>> updated_tracks = estimator.add_speed_and_distance_to_tracks(tracks)
        >>> print(updated_tracks['players'][0][1]['speed'])
        12.5  # Speed in km/h
    """
```

---

## 🛠️ Utility Functions

### `utils/parameter_scaler.py`

#### `scale_for_resolution(value, current_resolution, base_resolution=(1920, 1080))`

```python
def scale_for_resolution(value, current_resolution, base_resolution=(1920, 1080)):
    """
    Scale pixel-based parameters based on video resolution.
    
    Args:
        value (float): Base parameter value (calibrated at base_resolution)
        current_resolution (Tuple[int, int]): Current video resolution (width, height)
        base_resolution (Tuple[int, int]): Reference resolution (default 1920x1080)
    
    Returns:
        float: Scaled parameter value
        
    Examples:
        >>> # Scale for 720p video
        >>> scale_for_resolution(50.0, (1280, 720))
        33.333333333333336
        
        >>> # Scale for 4K video
        >>> scale_for_resolution(50.0, (3840, 2160))
        100.0
        
        >>> # Fallback if resolution is None
        >>> scale_for_resolution(50.0, None)
        50.0
    """
```

### `utils/bbox_utils.py`

#### `is_valid_bbox(bbox, frame_shape=None)`

```python
def is_valid_bbox(bbox, frame_shape=None):
    """
    Validate bounding box format and bounds.
    
    Args:
        bbox (List[float]): Bounding box [x1, y1, x2, y2]
        frame_shape (Tuple[int, int], optional): Frame dimensions (height, width)
        
    Returns:
        bool: True if bbox is valid, False otherwise
        
    Validation Checks:
        - Format: Must be list/tuple with 4 elements
        - Bounds: x1 < x2, y1 < y2
        - Frame bounds: Within frame dimensions
        - NaN values: No NaN or infinite values
        
    Example:
        >>> is_valid_bbox([100, 200, 150, 250])
        True
        >>> is_valid_bbox([150, 200, 100, 250])  # Invalid: x1 > x2
        False
    """
```

---

## 🛡️ Error Handling

### Error Types and Handling

#### 1. **Missing Detections**
```python
# In tracking_engine.py
if detection is None:
    tracks["players"].append({})
    tracks["referees"].append({})
    tracks["ball"].append({})
    continue
```

#### 2. **Invalid Bounding Boxes**
```python
# In bbox_utils.py
def is_valid_bbox(bbox, frame_shape=None):
    if bbox is None or len(bbox) != 4:
        return False
    if any(math.isnan(x) or math.isinf(x) for x in bbox):
        return False
    # ... additional validation
```

#### 3. **Lost Ball Frames**
```python
# In frame_processor.py
if ball_missing_count <= config['BALL_MAX_GAP_FRAMES'] and team_ball_control:
    team_ball_control.append(team_ball_control[-1])  # carry forward
else:
    team_ball_control.append(-1)  # unknown team
```

#### 4. **Memory Errors**
```python
# In motion_detector.py
try:
    # Process frames
    pass
except MemoryError:
    gc.collect()  # Force garbage collection
    # Retry with smaller batch
```

### Error Logging

```python
import logging

logger = logging.getLogger("football_analysis")

# Log errors with context
logger.error(f"Failed to process frame {frame_num}: {error}")
logger.warning(f"Low confidence detection: {confidence}")
logger.info(f"Successfully processed {frame_count} frames")
```

---

## 📊 Performance Metrics

### API Performance

| **Function** | **Complexity** | **Memory Usage** | **Optimization** |
|--------------|----------------|------------------|------------------|
| `get_object_tracks()` | O(n) | O(n) | ByteTrack caching |
| `detect_camera_movement()` | O(n) | O(1) | Feature downscaling |
| `get_player_team()` | O(1) | O(1) | Cached centroids |
| `smooth_speed_exponential()` | O(1) | O(1) | In-place smoothing |

### Memory Management

```python
# Automatic memory cleanup
import gc

# Force garbage collection after processing
gc.collect()

# Downscale frames for memory efficiency
old_gray = cv2.resize(old_gray_full, None, fx=scale, fy=scale)
```

---

## 🔧 Configuration Examples

### Basic Configuration
```python
config = {
    'MIN_PLAYERS_FOR_KMEANS': 6,
    'BALL_MAX_GAP_FRAMES': 15,
    'TEAM_CONFIDENCE_THRESHOLD': 0.6,
    'TEAM_HYSTERESIS_FRAMES': 5,
    'TEAM_MIN_COLOR_SEPARATION': 50.0,  # Auto-scaled
    'input_path': 'input_videos/video.mp4',
    'output_path': 'output_videos/result.avi'
}
```

### Advanced Configuration
```python
# Custom smoothing parameters
speed_estimator.smoothing_alpha = 0.5  # More responsive
speed_estimator.frame_window = 10     # Larger window

# Custom team assignment
team_assigner.confidence_threshold = 0.7  # Higher confidence
team_assigner.hysteresis_frames = 10      # More stable
```

---

## 🚀 Usage Examples

### Basic Usage
```python
from pipeline.config_manager import setup_configuration
from pipeline.module_initializer import initialize_modules

# Setup and run
config = setup_configuration()
modules = initialize_modules(config)

# Process video
tracks = modules['tracker'].get_object_tracks(frames)
movements = modules['camera_estimator'].detect_camera_movement(frames)
```

### Advanced Usage
```python
# Custom configuration
config = setup_configuration()
config['TEAM_CONFIDENCE_THRESHOLD'] = 0.8

# Initialize with custom config
modules = initialize_modules(config)

# Process with error handling
try:
    tracks = modules['tracker'].get_object_tracks(frames)
except Exception as e:
    logger.error(f"Tracking failed: {e}")
    tracks = {"players": [], "referees": [], "ball": []}
```

---

*This API documentation covers all major components of the Football Analysis System. For more details, see the source code and inline documentation.*

