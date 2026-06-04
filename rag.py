import sys
import chromadb
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import config
from utils import get_logger, seconds_to_hms, get_youtube_url

logger = get_logger("rag")

class PodcastRAGPipeline:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            api_key: Optional Gemini API key. If not provided, will load from config.py/dotenv.
        """
        # Load embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # Initialize ChromaDB client (persistent client)
        logger.info(f"Connecting to ChromaDB at: {config.CHROMA_DB_DIR}")
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        
        try:
            self.collection = self.chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)
            logger.info("ChromaDB collection successfully loaded.")
        except Exception as e:
            logger.warning(f"Could not load ChromaDB collection: {str(e)}. It might not be created yet.")
            self.collection = None

        # Set up Gemini API key
        self.api_key = api_key or config.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully.")
        else:
            logger.warning("No Gemini API key provided. LLM calls will fail until configured.")

    def set_api_key(self, api_key: str) -> None:
        """Dynamically set or update the Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        logger.info("Gemini API key updated dynamically.")

    def retrieve_chunks(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Embed the query and retrieve the top_k most similar chunks from ChromaDB.
        
        Returns:
            List of dictionaries containing document text, distance, similarity score,
            timestamp, formatted timestamp, and YouTube URL.
        """
        if not self.collection:
            # Try to fetch collection again in case it was built since initialization
            try:
                self.collection = self.chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)
            except Exception as e:
                logger.error("ChromaDB collection not initialized. Run embeddings.py first.")
                raise RuntimeError("Vector database has not been initialized. Please run embeddings.py.") from e

        logger.info(f"Retrieving top {top_k} chunks for query: '{query}'")
        
        # Embed user query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        
        # Check if we have results
        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            logger.warning("No matching documents found in ChromaDB.")
            return retrieved_chunks
            
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        
        for i in range(len(docs)):
            dist = distances[i]
            # Convert cosine distance to cosine similarity score
            # distance is 1 - similarity, so similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            
            meta = metadatas[i]
            retrieved_chunks.append({
                "id": ids[i],
                "text": docs[i],
                "distance": dist,
                "similarity_score": similarity,
                "start_time": meta.get("start_time", 0.0),
                "end_time": meta.get("end_time", 0.0),
                "formatted_start_time": meta.get("formatted_start_time", "00:00:00"),
                "youtube_url": meta.get("youtube_url", "")
            })
            
        return retrieved_chunks

    def answer_question(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> Dict[str, Any]:
        """
        Retrieve chunks, build prompt, call Gemini, and format the final answer payload.
        """
        if not self.api_key:
            raise ValueError("Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable or input it in the UI.")

        # 1. Retrieve relevant chunks
        chunks = self.retrieve_chunks(query, top_k=top_k)
        if not chunks:
            return {
                "answer": "This topic was not found in the podcast.",
                "confidence_score": 0.0,
                "timestamp_seconds": None,
                "formatted_timestamp": None,
                "youtube_url": None,
                "source_chunks": []
            }
            
        # 2. Extract best match timestamp (highest similarity score)
        best_chunk = max(chunks, key=lambda c: c["similarity_score"])
        
        # Average similarity of the retrieved context as general confidence
        avg_similarity = sum(c["similarity_score"] for c in chunks) / len(chunks)
        
        # 3. Format retrieved context for the prompt
        context_str = ""
        for idx, c in enumerate(chunks):
            context_str += f"[Chunk {idx+1}] (Timestamp: {c['formatted_start_time']})\n{c['text']}\n\n"
            
        # 4. Construct the prompt with strict guidelines
        prompt = f"""You are a precise and helpful assistant designed to answer questions about the podcast "Elon Musk × Nikhil Kamath | People by WTF Ep.16" using ONLY the transcript excerpts provided below.

Transcript Context:
---
{context_str}
---

Strict Constraints:
1. Answer the user's question based ONLY and EXCLUSIVELY on the Transcript Context provided above.
2. If the answer cannot be found in the Transcript Context, or if the context does not contain enough information to address the query, you MUST respond with this exact sentence:
"This topic was not found in the podcast."
3. Do NOT make assumptions, do NOT use external knowledge, and do NOT extrapolate. Never hallucinate.
4. Keep the answer complete but concise.

User Question: {query}
Answer:"""

        logger.info("Calling Gemini API...")
        try:
            model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            logger.info(f"Gemini API returned response length: {len(answer)}")
            
            # Check if Gemini output indicates it couldn't find the answer
            # We strip trailing punctuation or lower case it to be safe
            is_fallback = "This topic was not found in the podcast." in answer or answer.lower().startswith("this topic was not found")
            
            if is_fallback:
                return {
                    "answer": "This topic was not found in the podcast.",
                    "confidence_score": avg_similarity,
                    "timestamp_seconds": None,
                    "formatted_timestamp": None,
                    "youtube_url": None,
                    "source_chunks": chunks
                }
            
            return {
                "answer": answer,
                "confidence_score": best_chunk["similarity_score"],  # Best chunk similarity as match quality
                "timestamp_seconds": best_chunk["start_time"],
                "formatted_timestamp": best_chunk["formatted_start_time"],
                "youtube_url": best_chunk["youtube_url"],
                "source_chunks": chunks
            }
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise RuntimeError(f"Error generating answer from LLM: {str(e)}") from e

# Basic CLI test
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py \"<your question here>\"")
        sys.exit(0)
        
    question = sys.argv[1]
    try:
        pipeline = PodcastRAGPipeline()
        result = pipeline.answer_question(question)
        print("\n" + "="*50)
        print(f"Question: {question}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Timestamp: {result['formatted_timestamp']} ({result['timestamp_seconds']}s)")
        print(f"YouTube Link: {result['youtube_url']}")
        print(f"Answer:\n{result['answer']}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
