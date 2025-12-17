import streamlit as st
import ollama
import json
import re
import io
import time
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from sentence_transformers import CrossEncoder
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Agri-Crop Pro", page_icon="🌿", layout="wide")

# CUSTOM CSS for "App-like" feel
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 0; padding-bottom: 20px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# MODEL SETTINGS
# 'llama3.2-vision' is smarter but heavier. Use 'llava' if it crashes.
VISION_MODEL = "llava" 
CHAT_MODEL = "llava"
DB_PATH = "./vector_db"

# --- 2. INTELLIGENCE LAYER ---

@st.cache_resource
def load_rag():
    """Load the Knowledge Base."""
    # Ensure you have run 'ingest.py' first!
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return db, reranker

try:
    db, reranker = load_rag()
    RAG_READY = True
except:
    RAG_READY = False # Graceful fallback if no DB exists

def clean_json_output(text):
    """
    Robust JSON extractor. Fixes Markdown wraps and common LLM syntax errors.
    """
    try:
        # 1. Remove Markdown code blocks
        text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        
        # 2. Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0: return None
        
        json_str = text[start:end]
        
        # 3. Fix trailing commas (common error: "item", } -> "item" })
        json_str = re.sub(r",\s*}", "}", json_str)
        
        return json.loads(json_str)
    except:
        return None

def analyze_plant_health(image_bytes):
    """
    The 'Chain of Thought' Analysis Pipeline.
    """
    # PHASE 1: PURE VISION (Identify & Describe)
    # We ask it to just "See" first. No JSON yet.
    vision_prompt = """
    You are an expert Agronomist. Look at this image strictly.
    1. IDENTIFY the main object. Is it a plant? If so, what species? (e.g., Sunflower, Tomato).
    2. OBSERVE pathology. Are there spots, wilting, mold, or insects? 
    3. ASSESS severity.
    
    Output a concise paragraph description.
    """
    
    with st.status("🧠 Neural Vision Processing...", expanded=True) as status:
        st.write("📸 Scanning image structure...")
        vision_res = ollama.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': vision_prompt, 'images': [image_bytes]}]
        )
        description = vision_res['message']['content']
        st.write(f"**Detected:** {description[:100]}...")
        
        # PHASE 2: RAG SEARCH (The Library)
        context = "No manuals found."
        if RAG_READY:
            st.write("📚 Searching agricultural manuals...")
            results = db.similarity_search_with_score(description, k=5)
            doc_texts = [doc.page_content for doc, _ in results]
            
            # Re-rank based on the description
            scores = reranker.predict([[description, text] for text in doc_texts])
            top_docs = [text for score, text in sorted(zip(scores, doc_texts), reverse=True)[:3]]
            context = "\n".join(top_docs)
        
        # PHASE 3: SYNTHESIS (JSON Extraction)
        st.write("🔬 Formulating Diagnosis...")
        final_prompt = f"""
        Act as an AI Plant Doctor.
        
        IMAGE ANALYSIS: {description}
        REFERENCE MANUALS: {context}
        
        Task: Create a structured diagnosis.
        - If the image is NOT a plant (e.g., soil, tractor), set "is_plant": false.
        - If it is a plant, extract the Name, Disease, and Treatment.
        
        Return STRICT JSON format:
        {{
            "is_plant": true,
            "plant_name": "Name",
            "disease_name": "Name or 'Healthy'",
            "confidence": "High/Medium/Low",
            "description": "Simple explanation for a farmer.",
            "treatment": "Recommended action (Chemical/Organic)",
            "prevention": "Prevention tip"
        }}
        """
        
        final_res = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': final_prompt}])
        json_data = clean_json_output(final_res['message']['content'])
        
        status.update(label="✅ Complete", state="complete", expanded=False)
        return json_data, description

# --- 3. UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=50)
    st.header("Agri-Crop Pro")
    st.info("Upload a photo to detect diseases automatically.")
    
    uploaded_file = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'])
    
    st.divider()
    if RAG_READY:
        st.success("Database: Connected")
    else:
        st.warning("Database: Disconnected (Running in Vision-Only Mode)")

# MAIN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to inspect your crops."}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
    elif msg["role"] == "card":
        # RENDER CARD (Native Streamlit)
        data = msg["content"]
        with st.chat_message("assistant"):
            if not data.get("is_plant", True):
                st.error("⚠️ No plant detected. Please upload a clear photo of a leaf/fruit.")
            else:
                with st.container(border=True):
                    # Header
                    c1, c2 = st.columns([1, 1])
                    c1.metric("🌱 Plant", data.get("plant_name", "Unknown"))
                    c2.metric("🦠 Condition", data.get("disease_name", "Healthy"))
                    
                    st.divider()
                    st.markdown(f"**📝 Analysis:** {data.get('description')}")
                    
                    # Action Plan
                    st.success(f"**💊 Treatment:** {data.get('treatment')}")
                    st.info(f"**🛡️ Prevention:** {data.get('prevention')}")

# PROCESSING LOGIC
if uploaded_file and "last_file" not in st.session_state:
    st.session_state.last_file = uploaded_file.name
    
    # Show User Image
    image = Image.open(uploaded_file)
    st.session_state.messages.append({"role": "user", "content": "Analyzing upload..."})
    st.chat_message("user").image(image, width=250)
    
    # Run Analysis
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    
    data, debug_text = analyze_plant_health(img_bytes.getvalue())
    
    if data:
        st.session_state.messages.append({"role": "card", "content": data})
        st.rerun()
    else:
        st.error("Failed to parse diagnosis. The model might be hallucinating structure.")
        with st.expander("See Raw Output"):
            st.write(debug_text)

# Reset on new file
if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_name", ""):
    st.session_state.last_uploaded_name = uploaded_file.name
    del st.session_state.last_file
    st.rerun()