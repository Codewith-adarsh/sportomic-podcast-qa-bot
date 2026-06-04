import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

import config
from utils import get_logger, seconds_to_hms, get_youtube_url

logger = get_logger("embeddings")

def load_transcript(json_path: Path = config.TRANSCRIPT_JSON_PATH) -> List[Dict[str, Any]]:
    """Load the transcript from the cached JSON file."""
    if not json_path.exists():
        logger.error(f"Transcript file not found at {json_path}. Please run transcript.py first.")
        raise FileNotFoundError(f"Transcript file not found at {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_transcript(
    segments: List[Dict[str, Any]], 
    chunk_size_words: int = config.CHUNK_SIZE_WORDS, 
    overlap_words: int = config.CHUNK_OVERLAP_WORDS
) -> List[Dict[str, Any]]:
    """
    Groups transcript segments into coherent chunks of 400-600 words.
    Preserves the start timestamp of the first segment in the chunk,
    the end timestamp of the last segment, and implements a word overlap.
    
    Args:
        segments: Raw list of transcript segments with 'text', 'start', 'duration'.
        chunk_size_words: Target word size for each chunk.
        overlap_words: Word count overlap between consecutive chunks.
        
    Returns:
        List of chunks with text, start time, end time, and word count.
    """
    logger.info(f"Chunking transcript with target size={chunk_size_words} words, overlap={overlap_words} words.")
    
    chunks = []
    current_chunk_segments = []
    current_word_count = 0
    
    for seg in segments:
        text = seg['text']
        words = text.split()
        word_count = len(words)
        
        current_chunk_segments.append(seg)
        current_word_count += word_count
        
        # Once the current chunk reaches the word threshold, save it
        if current_word_count >= chunk_size_words:
            # Join text cleanly
            chunk_text = " ".join([s['text'] for s in current_chunk_segments])
            start_time = current_chunk_segments[0]['start']
            end_time = current_chunk_segments[-1]['start'] + current_chunk_segments[-1]['duration']
            
            chunks.append({
                "text": chunk_text,
                "start": start_time,
                "end": end_time,
                "word_count": current_word_count
            })
            
            # Slide window: backtrack to include overlap words in the next chunk
            overlap_segments = []
            overlap_count = 0
            for s in reversed(current_chunk_segments):
                s_words = len(s['text'].split())
                if overlap_count + s_words <= overlap_words:
                    overlap_segments.insert(0, s)
                    overlap_count += s_words
                else:
                    break
                    
            current_chunk_segments = overlap_segments
            current_word_count = overlap_count
            
    # Capture any trailing segments as the final chunk
    if current_chunk_segments and len(current_chunk_segments) > len(overlap_segments):
        chunk_text = " ".join([s['text'] for s in current_chunk_segments])
        start_time = current_chunk_segments[0]['start']
        end_time = current_chunk_segments[-1]['start'] + current_chunk_segments[-1]['duration']
        chunks.append({
            "text": chunk_text,
            "start": start_time,
            "end": end_time,
            "word_count": current_word_count
        })
        
    logger.info(f"Created {len(chunks)} text chunks from transcript.")
    return chunks

def build_vector_database() -> None:
    """
    Extracts chunks, generates embeddings using sentence-transformers,
    and indexes them in ChromaDB.
    """
    start_time = time.time()
    logger.info("Initializing Vector Database Build Process...")
    
    try:
        # Load transcript
        segments = load_transcript()
        
        # Chunk transcript
        chunks = chunk_transcript(segments)
        if not chunks:
            logger.warning("No chunks created from the transcript.")
            return
            
        # Initialize Sentence Transformer
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # Generate embeddings
        logger.info("Generating embeddings for all chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings successfully.")
        
        # Initialize ChromaDB Client
        logger.info(f"Initializing ChromaDB client at path: {config.CHROMA_DB_DIR}")
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        
        # Use Cosine similarity for better normalized scoring (1 - distance)
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clean existing items to allow re-indexing
        logger.info("Resetting/clearing existing database index for fresh ingest...")
        existing_items = collection.get()
        if existing_items and existing_items.get("ids"):
            collection.delete(ids=existing_items["ids"])
            logger.info(f"Deleted {len(existing_items['ids'])} existing chunks from database.")
            
        # Prepare metadata and IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        # Convert embeddings to standard floats list
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        metadatas = [
            {
                "start_time": c["start"],
                "end_time": c["end"],
                "word_count": c["word_count"],
                "formatted_start_time": seconds_to_hms(c["start"]),
                "youtube_url": get_youtube_url(c["start"])
            }
            for c in chunks
        ]
        
        # Ingest into ChromaDB
        logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB...")
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Vector Database built successfully in {elapsed:.2f} seconds!")
        
    except Exception as e:
        logger.error(f"Error during vector database build: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        build_vector_database()
        print("Vector Database created successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
