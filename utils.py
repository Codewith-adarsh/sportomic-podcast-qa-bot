import logging
import sys
from pathlib import Path
from config import LOG_FILE_PATH, YOUTUBE_VIDEO_ID, YOUTUBE_VIDEO_URL_BASE

def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a standard logger that writes logs to both 
    a file and stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def seconds_to_hms(seconds: float) -> str:
    """
    Convert a duration in seconds into HH:MM:SS format.
    
    Args:
        seconds: Float or int representation of time in seconds.
        
    Returns:
        A string formatted as HH:MM:SS.
    """
    try:
        total_seconds = int(round(float(seconds)))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "00:00:00"

def get_youtube_url(seconds: float, video_id: str = None) -> str:
    """
    Generate a clickable YouTube link that opens the video at the exact timestamp.
    
    Args:
        seconds: The timestamp in seconds.
        video_id: Optional video ID. Defaults to the one configured in config.py.
        
    Returns:
        A YouTube URL string with time parameter.
    """
    vid = video_id or YOUTUBE_VIDEO_ID
    try:
        secs_int = int(round(float(seconds)))
        # Keep secs_int non-negative
        secs_int = max(0, secs_int)
    except (ValueError, TypeError):
        secs_int = 0
    return f"{YOUTUBE_VIDEO_URL_BASE}{vid}&t={secs_int}s"
