import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from config import YOUTUBE_VIDEO_ID, TRANSCRIPT_JSON_PATH
from utils import get_logger

logger = get_logger("transcript")

def fetch_and_save_transcript(video_id: str = YOUTUBE_VIDEO_ID, output_path: Path = TRANSCRIPT_JSON_PATH) -> List[Dict[str, Any]]:
    """
    Fetch the transcript for a YouTube video and save it to a JSON file.
    
    Args:
        video_id: The 11-character YouTube video ID.
        output_path: Path to write the output transcript JSON.
        
    Returns:
        The transcript structure as a list of dictionaries with 'text', 'start', and 'duration'.
    """
    logger.info(f"Initiating transcript extraction for YouTube Video ID: {video_id}")
    
    try:
        # Instantiate the API client
        api = YouTubeTranscriptApi()
        
        # Fetch the transcript list
        transcript_list = api.list(video_id)
        
        # Try to get an English transcript (either manually created or auto-generated)
        try:
            transcript = transcript_list.find_transcript(['en'])
            logger.info("Found English transcript (either manual or auto-generated).")
        except NoTranscriptFound:
            # If English is not available, find the first available transcript and translate to English
            logger.info("No direct English transcript found. Attempting translation...")
            first_transcript = next(iter(transcript_list))
            transcript = first_transcript.translate('en')
            logger.info(f"Translated transcript from '{first_transcript.language}' to 'en'.")
            
        snippets = transcript.fetch()
        logger.info(f"Successfully fetched {len(snippets)} transcript segments.")
        
        # Convert the list of FetchedTranscriptSnippet objects to standard dictionaries
        data = [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            }
            for snippet in snippets
        ]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Transcript saved successfully to {output_path}")
        return data
        
    except TranscriptsDisabled as e:
        logger.error(f"Transcripts are disabled for video {video_id}: {str(e)}")
        raise RuntimeError(f"Transcripts are disabled for YouTube video: {video_id}") from e
    except NoTranscriptFound as e:
        logger.error(f"No transcript found for video {video_id}: {str(e)}")
        raise RuntimeError(f"No transcript found for YouTube video: {video_id}") from e
    except Exception as e:
        logger.error(f"Unexpected error while extracting transcript: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript from a YouTube video.")
    parser.add_argument("--video_id", type=str, default=YOUTUBE_VIDEO_ID, help="YouTube video ID to fetch transcript for")
    parser.add_argument("--output", type=str, default=str(TRANSCRIPT_JSON_PATH), help="Path to save the transcript JSON")
    
    args = parser.parse_args()
    
    try:
        fetch_and_save_transcript(video_id=args.video_id, output_path=Path(args.output))
        print(f"Success! Transcript saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
