"""
Parameter scaling utilities for dynamic threshold adaptation.

This module provides functions to scale parameters based on video characteristics
like resolution, ensuring thresholds work across different video qualities.
"""

def scale_for_resolution(value, current_resolution, base_resolution=(1920, 1080)):
    """
    Scale pixel-based parameters based on video resolution.
    
    This function scales parameters that are resolution-dependent (like pixel distances,
    color separation thresholds, etc.) to maintain consistent behavior across different
    video resolutions (720p, 1080p, 4K, etc.).
    
    Args:
        value: Base parameter value (calibrated at base_resolution)
        current_resolution: Tuple (width, height) of current video
        base_resolution: Reference resolution tuple (default 1920x1080)
    
    Returns:
        float: Scaled parameter value
        
    Examples:
        >>> # Scale for 720p video
        >>> scale_for_resolution(50.0, (1280, 720))
        33.33...
        
        >>> # Scale for 4K video
        >>> scale_for_resolution(50.0, (3840, 2160))
        100.0
        
        >>> # Fallback if resolution is None
        >>> scale_for_resolution(50.0, None)
        50.0
    """
    if current_resolution is None:
        return value  # Fallback to original value if resolution not available
    
    # Scale based on width ratio (most relevant for horizontal features)
    scale_factor = current_resolution[0] / base_resolution[0]
    return value * scale_factor

