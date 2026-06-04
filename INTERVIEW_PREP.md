# Internship Interview Preparation Guide: Podcast Q&A Bot

This guide compiles essential conceptual questions and detailed answers designed to help a fresher confidently explain the technical architecture, design decisions, and trade-offs of this Retrieval-Augmented Generation (RAG) system during an internship interview.

---

## 📖 Core RAG Concepts

### 1. What is Retrieval-Augmented Generation (RAG)?
**Answer:**
RAG is a technique that extends the capabilities of Large Language Models (LLMs) by retrieving relevant information from an external data source (like a database or document cache) and appending it to the LLM prompt as context. 
- **Retrieval:** When a user asks a question, the system queries a database (e.g., ChromaDB) to fetch chunks of text most related to the query.
- **Augmentation:** The retrieved text chunks are formatted and injected into the prompt along with the user's question.
- **Generation:** The LLM (e.g., Gemini 2.5 Flash) reads the retrieved context and generates a precise answer based *only* on that context.

### 2. Why use RAG instead of Fine-Tuning or prompting the LLM directly?
**Answer:**
| Criteria | Direct Prompting (Pre-trained) | Fine-Tuning | RAG (Our Solution) |
|---|---|---|---|
| **Data Recency** | Limited to training cutoff date. | Requires expensive retraining. | **Instant updates** (just replace/reindex files). |
| **Hallucination** | High risk (tends to make up facts). | Moderate risk. | **Very Low** (forced to rely on retrieved context). |
| **Auditability** | Cannot cite source or verify. | Hard to trace weights. | **Direct citations** (provides exact timestamps/text). |
| **Compute Cost** | Low. | High (GPU compute required). | **Extremely Low** (uses standard database queries). |

---

## 🧮 Embeddings & Semantic Search

### 3. What are Text Embeddings and why do we represent text as vectors?
**Answer:**
Text embeddings are numerical representations of text segments. An embedding model (like `all-MiniLM-L6-v2`) converts a text string into a high-dimensional vector (a list of numbers, 384 dimensions for our model).
- These vectors are trained such that words or sentences with **similar semantic meanings** are located close to each other in the vector space.
- Computers cannot understand language syntax and nuances, but they are exceptionally fast at calculating distance in mathematical coordinate spaces. Converting text to vectors allows us to perform mathematical similarity searches.

### 4. What is the difference between Keyword Search (Lexical) and Semantic Search?
**Answer:**
- **Keyword Search (e.g., BM25, SQL LIKE):** Looks for exact text matches. If a user asks *"Is there life beyond Earth?"* and the transcript says *"We have not found extraterrestrial biological entities,"* keyword search will fail because "Earth," "life," and "extraterrestrial" do not match.
- **Semantic Search (Vector Search):** Matches based on conceptual meaning. Because the embedding model understands that " extraterrestrial biological entities" is conceptually related to "life beyond Earth," it will return a high similarity score and successfully retrieve the passage.

---

## 🗄️ Vector Databases

### 5. Why did we use ChromaDB? Why not a relational database like MySQL?
**Answer:**
ChromaDB is a specialized **vector database**. 
- Relational databases are designed to index tabular data (integers, strings, dates) using B-trees. They cannot efficiently index or compare high-dimensional floating-point vectors.
- ChromaDB is optimized for **Approximate Nearest Neighbor (ANN)** search algorithms (like HNSW). It allows us to compare a query vector against thousands of stored vectors and return the top-K matches in milliseconds.
- ChromaDB is also **embedded and serverless**; it runs directly inside the Python process and writes data locally to disk, making it extremely easy to set up and deploy without managing external server infrastructure.

### 6. How is the "Match Confidence Score" calculated?
**Answer:**
We configured our ChromaDB collection to use **Cosine Similarity** (`hnsw:space: cosine`). 
- Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 to 1.
- In ChromaDB, the returned "distance" is **Cosine Distance**, which is mathematically defined as `1 - Cosine Similarity`.
- To convert this back to a reader-friendly confidence score, we calculate:
  $$\text{Similarity Score} = 1.0 - \text{Cosine Distance}$$
  We clip this score between `0.0` (no match) and `1.0` (perfect match) to ensure robust percentage representations in our UI.

---

## ⏱️ Timestamp Mapping System

### 7. How does the system map generated answers to exact timestamps in the YouTube video?
**Answer:**
1. **Extraction:** When `transcript.py` runs, the `youtube-transcript-api` returns segments containing the starting offset in seconds (e.g. `start: 125.4` seconds).
2. **Persistence:** During chunking (in `embeddings.py`), we group segments into 500-word blocks. We extract the `start` time of the *first* segment in the block and store it as a metadata field (`start_time`) alongside the chunk inside ChromaDB.
3. **Retrieval:** When a question is asked, ChromaDB returns the top-4 chunks. We isolate the metadata of the chunk with the highest similarity score.
4. **Link Generation:** In `utils.py`, we convert the `start_time` float (e.g., `125`) to an integer and build the URL: `https://youtube.com/watch?v=VIDEO_ID&t=125s`. We also format the time to `HH:MM:SS` (e.g., `00:02:05`) using standard arithmetic for UI display.

---

## ⚠️ System Limitations & Improvements

### 8. What are the main limitations of the current system?
**Answer:**
1. **Single Source Limitation:** It is locked to one YouTube video transcript. Searching across a playlist or channel would require an upgraded database structure.
2. **Context Size Limitation:** Standard RAG injects the top-4 chunks. If a question requires scanning the entire 2-hour podcast to calculate an answer (e.g., *"How many times did they mention SpaceX?"*), standard RAG will fail because it only sees local segments.
3. **API Dependency:** Gemini 2.5 Flash requires an active internet connection and API key. If the key expires or internet is cut, answer generation fails.

### 9. How would you upgrade this system for production scale? (Future Work)
**Answer:**
1. **Hybrid Search:** Combine keyword search (BM25) with vector search (Semantic) using Reciprocal Rank Fusion (RRF). This ensures exact terms (like names of companies or specific codes) are retrieved alongside conceptual matches.
2. **Reranking:** Fetch the top-20 chunks from ChromaDB and pass them through a cross-encoder model (like Cohere Rerank). Rerankers are highly accurate at sorting passages by query relevance, ensuring the best context enters the LLM.
3. **Query Expansion/Rewriting:** Use an LLM to generate multiple search variations of the user's question, run them in parallel, and merge results. This improves recall.
4. **Agentic RAG:** Implement routing logic so that if the query requires counting, summarization, or synthesis across the whole transcript, the agent uses code execution or map-reduce steps instead of standard lookup.
5. **RAG Evaluation (Ragas/TruLens):** Implement automated frameworks to measure context precision, context recall, faithfulness (checking if LLM hallucinated outside context), and answer relevance.
