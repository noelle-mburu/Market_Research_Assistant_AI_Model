import os
import re
import tempfile
import requests
import base64
import streamlit as st
from streamlit_chat import message
import speech_recognition as sr
from gtts import gTTS
import matplotlib.pyplot as plt
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# CONFIG
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("❌ 'GROQ_API_KEY' is not set in environment variables.")
    st.stop()

THEME_COLORS = {"bg": "#000000", "accent": "#00FF7F", "muted": "#111111"}
ENABLE_TTS = True

# CSS Styles
CSS_STYLE = """
    <style>
    .stApp {
        background: #000000;
        color: #00FF7F;
    }
    .stTextInput, .stButton>button {
        background-color: #111111 !important;
        color: #00FF7F !important;
        border-radius: 8px;
        border: 1px solid #00FF7F;
    }
    .voice-button {
        background: linear-gradient(45deg, #00FF7F, #00CC66) !important;
        color: black !important;
        font-weight: bold !important;
    }
    .tts-button {
        background: linear-gradient(45deg, #FF6B00, #FF8C00) !important;
        color: black !important;
        font-weight: bold !important;
        margin-top: 10px;
    }
    </style>
"""

# =============================
# SESSION STATE
# =============================
if "agent" not in st.session_state:
    st.session_state.agent = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_sources" not in st.session_state:
    st.session_state.data_sources = {"pdf": [], "csv": [], "web": []}
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


# VOICE INPUT FUNCTION
# =============================
def get_voice_input():
    """Capture voice input using microphone"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

        text = recognizer.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "Voice input timeout. Please try again."
    except sr.UnknownValueError:
        return "Could not understand audio. Please try again."
    except Exception as e:
        return f"Voice input error: {str(e)}"


# TEXT-TO-SPEECH FUNCTION
# =============================
def text_to_speech(text, lang='en'):
    """Convert text to speech and return audio data"""
    try:
        # Clean text for TTS (remove markdown, special characters, etc.)
        clean_text = re.sub(r'[\*_`#\[\]]', '', text)  # Remove markdown
        clean_text = re.sub(r'http\S+', '', clean_text)  # Remove URLs
        clean_text = clean_text.strip()

        if len(clean_text) < 10:  # Skip very short responses
            return None

        tts = gTTS(text=clean_text[:1000], lang=lang, slow=False)  # Limit text length
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None


# DATA VALIDATION
# =============================
def has_valid_data():
    """Check if we have valid data sources (not empty or invalid)"""
    # Check PDFs
    pdfs = st.session_state.data_sources.get("pdf", [])
    if pdfs:
        for pdf in pdfs:
            if pdf and hasattr(pdf, 'size') and pdf.size > 0:
                return True

    # Check if vector database has documents
    if (st.session_state.agent and
        st.session_state.agent.vector_db and
        hasattr(st.session_state.agent.vector_db, '_collection')):
        doc_count = st.session_state.agent.vector_db._collection.count()
        if doc_count > 0:
            return True

    return False


# AGENT
# =============================
class MarketResearchAgent:
    def __init__(self):
        # Fix: Use device='cpu' to avoid meta tensor issues
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_db = None
        self.qa_chain = None

    def create_vector_db(self):
        """Initialize or load a Chroma DB"""
        if self.vector_db is None:
            os.makedirs("./chroma_db", exist_ok=True)
            self.vector_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
        return self.vector_db

    def load_documents(self):
        docs = []
        pdfs = st.session_state.data_sources.get("pdf", [])

        for pdf_file in pdfs:
            # Validate PDF file
            if not pdf_file or (hasattr(pdf_file, 'size') and pdf_file.size == 0):
                continue

            try:
                if hasattr(pdf_file, 'read'):
                    pdf_file.seek(0)
                    reader = PyPDF2.PdfReader(pdf_file)
                else:
                    reader = PyPDF2.PdfReader(open(pdf_file, 'rb'))

                for i, page in enumerate(reader.pages):
                    txt = page.extract_text()
                    if txt and txt.strip():
                        docs.append(Document(
                            page_content=txt.strip(),
                            metadata={"source": getattr(pdf_file, 'name', str(pdf_file)), "page": i + 1}
                        ))
            except Exception as e:
                st.error(f"PDF read error: {e}")
        return docs

    def add_pdfs_to_db(self):
        docs = self.load_documents()
        if not docs:
            return False

        self.create_vector_db()
        split_docs = self.text_splitter.split_documents(docs)
        if split_docs:
            self.vector_db.add_documents(split_docs)
            self.vector_db.persist()
            return True
        return False

    def search_web(self, query, max_results=3):
        try:
            search_results = []
            with DDGS() as ddgs:
                for r in list(ddgs.text(query, max_results=max_results)):
                    try:
                        resp = requests.get(r["href"], timeout=10)
                        soup = BeautifulSoup(resp.text, "html.parser")
                        text = re.sub(r"\s+", " ", soup.get_text()).strip()[:5000]
                        if text:  # Only add if we got actual content
                            search_results.append({
                                "title": r["title"],
                                "url": r["href"],
                                "content": text
                            })
                    except:
                        continue

            if not search_results:
                return []

            self.create_vector_db()
            for r in search_results:
                doc = Document(
                    page_content=r["content"],
                    metadata={
                        "source": r["url"],
                        "title": r["title"],
                        "date": datetime.now().isoformat()
                    }
                )
                split_docs = self.text_splitter.split_documents([doc])
                if split_docs:
                    self.vector_db.add_documents(split_docs)

            self.vector_db.persist()
            return search_results

        except Exception as e:
            st.error(f"Web search error: {e}")
            return []

    def setup_qa_chain(self):
        self.create_vector_db()
        if self.vector_db is None:
            return None

        retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            temperature=0,
            model_name="llama-3.1-8b-instant"
        )

        try:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.conversation_memory,
                return_source_documents=True,
                output_key='answer',
                verbose=True
            )
            return self.qa_chain
        except Exception as e:
            st.error(f"Chain setup error: {e}")
            from langchain.chains import RetrievalQA
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            return self.qa_chain

    def process_query(self, query):
        # Check if we have valid data first
        has_data = has_valid_data()

        # Auto-search for recent/trend queries
        if any(k in query.lower() for k in ["recent", "latest", "2024", "2025", "trend", "current"]):
            results = self.search_web(query)
            if results:
                has_data = True

        # Add PDFs if available and valid
        pdfs = st.session_state.data_sources.get("pdf", [])
        if pdfs:
            pdf_added = self.add_pdfs_to_db()
            if pdf_added:
                has_data = True

        # Initialize vector DB if not exists
        if self.vector_db is None:
            self.create_vector_db()

        # Final check for data existence
        if not has_data:
            # Use LLM directly without retrieval
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                temperature=0,
                model_name="llama-3.1-8b-instant"
            )
            response = llm.invoke(f"As a gaming market research expert, answer based on your knowledge: {query}")
            return response.content, []

        # Setup QA chain if not exists
        if self.qa_chain is None:
            self.setup_qa_chain()

        if self.qa_chain is None:
            return "Error: Could not initialize the QA system.", []

        try:
            if hasattr(self.qa_chain, 'invoke'):
                result = self.qa_chain.invoke({"question": query})
            else:
                result = self.qa_chain({"question": query})

            if hasattr(result, 'get'):
                answer = result.get("answer", result.get("result", "No answer generated."))
                sources = result.get("source_documents", [])
            else:
                answer = str(result)
                sources = []

            return answer, sources
        except Exception as e:
            return f"Error processing query: {str(e)}", []



# APP
# =============================
def main():
    st.set_page_config(page_title="🎮 Gaming Market Research AI", layout="wide")

    # Apply custom CSS
    st.markdown(CSS_STYLE, unsafe_allow_html=True)

    st.title("🎮 Gaming-Style AI Chatbot")
    st.markdown("### Voice-enabled Market Research Assistant")

    # File upload section
    st.sidebar.header("📁 Data Sources")
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        # Filter out empty/invalid files
        valid_pdfs = [pdf for pdf in uploaded_pdfs if pdf and hasattr(pdf, 'size') and pdf.size > 0]
        st.session_state.data_sources["pdf"] = valid_pdfs
        if valid_pdfs:
            st.sidebar.success(f"Loaded {len(valid_pdfs)} valid PDF file(s)")
        else:
            st.sidebar.warning("No valid PDF files found")

    # Web search section
    st.sidebar.header("🌐 Web Search")
    search_query = st.sidebar.text_input("Search for recent gaming trends")
    if st.sidebar.button("Search Web"):
        if search_query and search_query.strip():
            with st.spinner("Searching web..."):
                if st.session_state.agent is None:
                    st.session_state.agent = MarketResearchAgent()
                results = st.session_state.agent.search_web(search_query)
                if results:
                    st.sidebar.success(f"Found {len(results)} results")
                else:
                    st.sidebar.error("No results found")

    # TTS Settings
    st.sidebar.header("🔊 Speech Settings")
    enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=True, help="Read responses aloud")

    if st.session_state.agent is None:
        st.session_state.agent = MarketResearchAgent()
    agent = st.session_state.agent

    # Display current data sources
    if has_valid_data():
        st.sidebar.success("✅ Data sources loaded and ready!")
    else:
        st.sidebar.info("📝 Add PDFs or web search to enhance responses")

    # Chat history
    st.markdown("#### 💬 Conversation")
    
    if not st.session_state.messages:
        st.info("💡 **Tips:** Upload PDFs or use web search to enhance responses with actual data!")

    for i, chat in enumerate(st.session_state.messages):
        if chat["role"] == "user":
            message(chat["content"], is_user=True, key=f"user_{i}")
        else:
            message(chat["content"], is_user=False, key=f"assistant_{i}")

            # Add TTS button for each assistant response
            if enable_tts and i == len(st.session_state.messages) - 1:  # Only for latest response
                if st.button("🔊 Read Aloud", key=f"tts_{i}", use_container_width=True, type="secondary"):
                    audio_data = text_to_speech(chat["content"])
                    if audio_data:
                        st.session_state.last_audio = audio_data
                        st.audio(audio_data, format="audio/mp3")

    # User input section with voice input
    st.markdown("#### 💬 Ask a Question")

    # Create columns for voice input and text input
    col1, col2 = st.columns([4, 1])
    
    # Voice input button
    with col2:
        if st.button("🎤 Voice Input", key="voice_btn", use_container_width=True, type="primary"):
            voice_text = get_voice_input()
            if voice_text and "error" not in voice_text.lower() and "timeout" not in voice_text.lower():
                st.session_state.voice_input = voice_text
                st.success("Voice captured!")

    # Display voice input if available
    if st.session_state.voice_input:
        st.info(f"🎤 **Voice input:** {st.session_state.voice_input}")

    # Text input
    with col1:
        user_input = st.chat_input("Type your gaming market research question...")

    # Use voice input if available, otherwise use text input
    final_input = ""
    if st.session_state.voice_input:
        final_input = st.session_state.voice_input
        # Don't clear immediately - let processing handle it
    elif user_input:
        final_input = user_input

    # Process query - only if we have new input and not currently processing
    if final_input and final_input.strip() and not st.session_state.processing:
        # Set processing flag to prevent re-processing
        st.session_state.processing = True
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": final_input})
        
        # Clear voice input after adding to messages
        if st.session_state.voice_input:
            st.session_state.voice_input = ""
        
        with st.spinner("🔍 Analyzing gaming market data..."):
            response, sources = agent.process_query(final_input)

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Generate TTS audio if enabled
        if enable_tts:
            audio_data = text_to_speech(response)
            if audio_data:
                st.session_state.last_audio = audio_data
                # Auto-play the audio (note: autoplay may be blocked by browsers)
                st.audio(audio_data, format="audio/mp3")
        
        # Reset processing flag
        st.session_state.processing = False
        
        # Use rerun to refresh the display
        st.rerun()


    # Display visualization in sidebar
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        latest_response = st.session_state.messages[-1]["content"]

        # Try to extract numbers for visualization
        nums = re.findall(r'\d+', latest_response)
        if len(nums) >= 2:
            st.sidebar.header("📊 Trend Visualization")
            try:
                nums = list(map(int, nums[:6]))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(nums, marker="o", color="#00FF7F", linewidth=2)
                ax.grid(True, alpha=0.3)
                ax.set_title("Detected Numerical Trend", color="#00FF7F")
                ax.set_facecolor("#000000")
                fig.patch.set_facecolor('#000000')
                ax.tick_params(colors='#00FF7F')
                st.sidebar.pyplot(fig)
            except Exception as e:
                st.sidebar.info("Visualization available when numbers are detected")

if __name__ == "__main__":
    main()