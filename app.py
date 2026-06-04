import streamlit as st
from pathlib import Path
import os

from rag import PodcastRAGPipeline
import config
from utils import get_logger

logger = get_logger("app")

# Page Configuration
st.set_page_config(
    page_title="WTF Podcast Q&A Bot - Elon Musk & Nikhil Kamath",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main container and font settings */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 2rem;
    }
    
    /* Elegant Cards */
    .answer-card {
        background-color: #172a45;
        border-left: 5px solid #00f2fe;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        color: #ccd6f6;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    .timestamp-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid #305261;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .timestamp-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #00f2fe;
    }
    
    .source-chunk-card {
        background-color: #0a192f;
        border: 1px solid #233554;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .source-chunk-card:hover {
        border-color: #00f2fe;
        transform: translateY(-2px);
    }
    
    .chunk-header {
        display: flex;
        justify-content: space-between;
        color: #8892b0;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #172a45;
        padding-bottom: 0.3rem;
    }
    
    .chunk-text {
        color: #ccd6f6;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Button Links */
    .youtube-btn {
        background-color: #ff0000;
        color: white !important;
        padding: 0.6rem 1.2rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        transition: background-color 0.3s ease;
    }
    
    .youtube-btn:hover {
        background-color: #cc0000;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 50px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    
    .status-active {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    
    .status-missing {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Cache pipeline globally
@st.cache_resource
def initialize_pipeline() -> PodcastRAGPipeline:
    """Initialize and cache the RAG pipeline."""
    return PodcastRAGPipeline()

# Try to initialize the cached pipeline
try:
    pipeline = initialize_pipeline()
except Exception as e:
    st.error(f"Error loading Vector Database: {e}")
    st.info("Make sure you have run transcript.py and embeddings.py to setup files.")
    pipeline = None

# Sidebar Configuration
with st.sidebar:
    st.title("🎙️ Control Panel")
    
    # 1. Check API Key
    st.subheader("🔑 Gemini API Settings")
    api_key_env = os.getenv("GEMINI_API_KEY")
    
    user_api_key = st.text_input(
        "Enter Gemini API Key:",
        type="password",
        value=api_key_env or "",
        help="Get an API key from Google AI Studio. Stored in memory only."
    )
    
    if user_api_key:
        if pipeline:
            pipeline.set_api_key(user_api_key)
        st.success("API Key applied!")
    elif not api_key_env:
        st.warning("Please provide a Gemini API Key to make queries.")
        
    st.markdown("---")
    
    # 2. System Status Checks
    st.subheader("⚙️ System Status")
    
    transcript_exists = config.TRANSCRIPT_JSON_PATH.exists()
    db_exists = config.CHROMA_DB_DIR.exists() and len(os.listdir(config.CHROMA_DB_DIR)) > 0 if config.CHROMA_DB_DIR.exists() else False
    api_key_configured = bool(user_api_key or api_key_env)
    
    def render_status(label: str, active: bool):
        badge_class = "status-active" if active else "status-missing"
        badge_text = "Ready" if active else "Missing"
        st.markdown(f"**{label}**: <span class='status-badge {badge_class}'>{badge_text}</span>", unsafe_allow_html=True)

    render_status("Transcript Cache", transcript_exists)
    render_status("ChromaDB Vector Index", db_exists)
    render_status("Gemini API Connection", api_key_configured)
    
    if not (transcript_exists and db_exists):
        st.info("💡 To initialize the database, execute `python transcript.py` followed by `python embeddings.py` in your terminal.")
        
    st.markdown("---")
    
    # 3. Example Questions List
    st.subheader("💡 Example Questions")
    st.markdown("Click any question to load it:")
    
    example_questions = [
        "What is Elon Musk's view on artificial intelligence?",
        "What is discussed regarding the colonization of Mars?",
        "What does Elon Musk say about the future of work and jobs?",
        "Does Elon Musk believe in aliens or extraterrestrial life?",
        "How does Nikhil Kamath introduce the guest and set the context?",
        "What are Elon's views on the educational system?",
        "How does Elon Musk view demographic declines and birth rates?",
        "What is discussed about Neuralink and its goals?",
        "What is Elon's perspective on regulatory frameworks for AI?",
        "What does Elon Musk say about China and its engineering talent?"
    ]
    
    # Click handler for example questions using Session State
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
        
    for idx, q in enumerate(example_questions):
        if st.button(f"{idx+1}. {q}", key=f"q_btn_{idx}", help=f"Ask: {q}"):
            st.session_state.question_input = q
            st.rerun()

# Main Application Layout
st.markdown("<h1 class='main-title'>WTF Podcast Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instant Semantic Search & AI Answer Synthesis for: <b>Elon Musk × Nikhil Kamath | People by WTF Ep.16</b></p>", unsafe_allow_html=True)

# Display Banner Image
banner_path = config.DATA_DIR / "banner.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)
else:
    st.warning("Banner image not found. Place your banner image in data/banner.png.")

# Question Section
st.markdown("### 🔍 Search the Podcast")

# Text Area linked to Session State
user_query = st.text_area(
    "What topic would you like to search for?",
    value=st.session_state.question_input,
    placeholder="Example: How does Elon feel about the education system? or What is Neuralink's core objective?",
    height=80,
    key="user_query_input"
)

# Sync manual typing back to session state to prevent loss on rerun
st.session_state.question_input = user_query

col1, col2 = st.columns([1, 6])
with col1:
    ask_button = st.button("Ask Bot 🚀", use_container_width=True)

# Process Question
if ask_button:
    if not user_query.strip():
        st.error("Please enter a question or click one of the example questions in the sidebar.")
    elif not pipeline:
        st.error("RAG pipeline failed to initialize. Please ensure ChromaDB contains data.")
    elif not (user_api_key or api_key_env):
        st.error("Missing Gemini API Key. Please input your API key in the sidebar control panel.")
    else:
        # Prepare pipeline key dynamic check
        current_api_key = user_api_key or api_key_env
        pipeline.set_api_key(current_api_key)
        
        # Display Loading States
        with st.status("🔍 Analyzing podcast database & synthesizing answer...", expanded=True) as status_box:
            
            status_box.write("1. Connecting to ChromaDB & computing query embedding...")
            # Run semantic search
            try:
                retrieved_chunks = pipeline.retrieve_chunks(user_query)
                status_box.write(f"2. Found {len(retrieved_chunks)} relevant transcript segments (semantic match).")
            except Exception as e:
                status_box.update(state="error", label=f"ChromaDB retrieval failed: {e}")
                st.stop()
                
            status_box.write("3. Compiling context and executing Gemini RAG generation...")
            try:
                result = pipeline.answer_question(user_query)
                status_box.update(state="complete", label="Processing completed successfully!")
            except Exception as e:
                status_box.update(state="error", label=f"Gemini generation failed: {e}")
                st.stop()
                
        # RENDER RESULTS
        st.markdown("### 🤖 Synthesized Answer")
        
        # 1. Answer Card
        st.markdown(f"<div class='answer-card'>{result['answer']}</div>", unsafe_allow_html=True)
        
        # 2. Check if answer was found (i.e. did not trigger fallback)
        if result["youtube_url"] and result["formatted_timestamp"]:
            st.markdown("### ⏱️ Timestamp & Video Link")
            
            # Formulate similarity text / confidence
            confidence = result["confidence_score"]
            color = "#2ecc71" if confidence > 0.75 else ("#f39c12" if confidence > 0.55 else "#e74c3c")
            
            # Render Timestamp Card
            st.markdown(f"""
            <div class='timestamp-card'>
                <div>
                    <span style='font-size: 1.1rem; color: #8892b0;'>Discussion Timestamp:</span>
                    <span class='timestamp-text'>&nbsp;{result['formatted_timestamp']}</span>
                    <span style='margin-left: 1.5rem; font-size: 0.95rem; color: #8892b0;'>
                        Match Confidence: <b style='color: {color};'>{confidence:.1%}</b>
                    </span>
                </div>
                <a class='youtube-btn' href='{result['youtube_url']}' target='_blank'>
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" style="display:inline-block; vertical-align:middle;">
                        <path d="M23.498 6.163a3.003 3.003 0 0 0-2.11-2.11C19.518 3.545 12 3.545 12 3.545s-7.518 0-9.388.508a3.003 3.003 0 0 0-2.11 2.11C0 8.033 0 12 0 12s0 3.967.502 5.837a3.003 3.003 0 0 0 2.11 2.11C4.482 20.455 12 20.455 12 20.455s7.518 0 9.388-.508a3.003 3.003 0 0 0 2.11-2.11C24 15.967 24 12 24 12s0-3.967-.502-5.837zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                    </svg>
                    Play Video at Timestamp
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual confidence progress bar
            st.progress(confidence)
            
        else:
            st.warning("⚠️ No exact timestamp matches because this topic was not found in the podcast.")
            
        # 3. Source Context Expander
        if result["source_chunks"]:
            with st.expander("📄 Source Transcript Context (Semantic Search Results)", expanded=False):
                st.markdown("Below are the raw transcript excerpts retrieved from ChromaDB that were used by the LLM as context:")
                
                for idx, chunk in enumerate(result["source_chunks"]):
                    chunk_conf = chunk["similarity_score"]
                    chunk_color = "#2ecc71" if chunk_conf > 0.75 else ("#f39c12" if chunk_conf > 0.55 else "#e74c3c")
                    
                    st.markdown(f"""
                    <div class='source-chunk-card'>
                        <div class='chunk-header'>
                            <span><b>Exercept #{idx+1}</b> | ID: {chunk['id']}</span>
                            <span>Starts at: <b>{chunk['formatted_start_time']}</b> | Similarity: <b style='color: {chunk_color};'>{chunk_conf:.1%}</b></span>
                        </div>
                        <div class='chunk-text'>{chunk['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer Info
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #57606f; font-size: 0.85rem;'>"
    "Sportomic AI Lab Internship Assignment - Built with ❤️ by an AI Engineer Candidate<br>"
    "Tech Stack: Python, Streamlit, ChromaDB, Sentence Transformers, Gemini 2.5 Flash, YouTube Transcript API"
    "</p>",
    unsafe_allow_html=True
)
