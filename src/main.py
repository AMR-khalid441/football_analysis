import logging
from pipeline.config_manager import setup_configuration
from pipeline.module_initializer import initialize_modules
from pipeline.video_setup import setup_video_io
from pipeline.team_initializer import initialize_team_colors
from pipeline.frame_processor import process_video_frames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("football_analysis")

def main():
    # 1. Setup configuration
    config = setup_configuration()
    
    # 2. Initialize modules
    modules = initialize_modules(config)
    
    # 3. Setup video I/O
    video_gen, first_frame, out_writer = setup_video_io(config['input_path'], config['output_path'])
    
    # 4. Initialize team colors
    initialize_team_colors(modules['tracker'], modules['team_assigner'], modules['camera_estimator'], first_frame, config)
    
    # 5. Process video
    process_video_frames(video_gen, first_frame, modules, config, out_writer)

if __name__ == "__main__":
    main()
