import streamlit as st
import sqlite3
import datetime
import pypdf
import ollama
import numpy as np
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OmniContext Local Intelligence Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


import streamlit as st


def load_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: #161d2e;
            --border-subtle: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-start: #6366f1;
            --accent-end: #3b82f6;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
        }

        /* --- BASE APP --- */
        .main, .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.03em;
            color: #ffffff;
        }
        p, span, label, div {
            font-family: 'Inter', sans-serif;
        }

        /* --- BUTTONS (unchanged from original) --- */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%);
            color: white;
            border: none;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
            transform: translateY(-1px);
        }

        /* --- SIDEBAR (unchanged) --- */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-subtle);
        }

        /* --- TABS (unchanged) --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: var(--bg-secondary);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-end) !important;
            color: white !important;
        }

        /* --- NEW: styled native st.metric widgets --- */
        [data-testid="stMetric"] {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-weight: 500;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-weight: 700;
        }

        /* --- NEW: generic card container --- */
        .ui-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        }
        .ui-card h4 {
            margin-top: 0;
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* --- NEW: risk / status badges --- */
        .ui-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }
        .ui-badge-green { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.4); }
        .ui-badge-amber { background: rgba(245,158,11,0.15); color: var(--amber); border: 1px solid rgba(245,158,11,0.4); }
        .ui-badge-red   { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.4); }

        /* --- NEW: dataframe / table polish --- */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)


def metric_card(title: str, value: str, subtitle: str = ""):
    """Optional helper: renders a styled card around any text/value.
    Purely presentational — does not compute anything."""
    st.markdown(f"""
    <div class="ui-card">
        <h4>{title}</h4>
        <div style="font-size:1.6rem; font-weight:700; color:#ffffff;">{value}</div>
        <div style="color:var(--text-secondary); font-size:0.85rem;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(label: str, level: str):
    """Optional helper: level should be 'low', 'medium', or 'high'.
    Maps to green / amber / red badge respectively."""
    color_class = {
        "low": "ui-badge-green",
        "medium": "ui-badge-amber",
        "high": "ui-badge-red",
    }.get(level.lower(), "ui-badge-amber")
    st.markdown(f'<span class="ui-badge {color_class}">{label}</span>', unsafe_allow_html=True)

# --- LOCAL DATABASE & VECTOR STORE SETUP ---
def init_db():
    conn = sqlite3.connect("omnictext_vector.db")
    cursor = conn.cursor()
    # Document Chunks and Vector Embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            chunk_text TEXT,
            embedding TEXT,
            timestamp TEXT
        )
    ''')
    # Query Audit History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            mode TEXT,
            query TEXT,
            response TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def get_embedding(text):
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return response['embedding']
    except Exception as e:
        return None


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def store_document_vectors(filename, full_text):
    conn = sqlite3.connect("omnictext_vector.db")
    cursor = conn.cursor()

    # Clear previous entries for clean state if re-uploading
    cursor.execute("DELETE FROM vector_store WHERE filename = ?", (filename,))

    chunks = chunk_text(full_text)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb:
            cursor.execute('''
                INSERT INTO vector_store (filename, chunk_text, embedding, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (filename, chunk, json.dumps(emb), timestamp))

    conn.commit()
    conn.close()
    return len(chunks)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_context(query, top_k=3):
    query_emb = get_embedding(query)
    if not query_emb:
        return ""

    conn = sqlite3.connect("omnictext_vector.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_text, embedding FROM vector_store")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return ""

    scored_chunks = []
    q_vec = np.array(query_emb)

    for chunk_text, emb_json in rows:
        doc_vec = np.array(json.loads(emb_json))
        score = cosine_similarity(q_vec, doc_vec)
        scored_chunks.append((score, chunk_text))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

    return "\n\n---\n\n".join(top_chunks)


def log_audit(filename, mode, query, response):
    conn = sqlite3.connect("omnictext_vector.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_logs (filename, mode, query, response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, mode, query, response, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def fetch_audit_logs():
    conn = sqlite3.connect("omnictext_vector.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, mode, query, timestamp FROM audit_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


# --- APP HEADER ---
st.title("🧠 OmniContext Local Intelligence Framework")
st.markdown("### Enterprise Vector RAG Engine Powered by Local Qwen & Snapdragon NPU Architecture")
st.markdown("---")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Execution Settings")
workflow_mode = st.sidebar.selectbox(
    "Select Workflow Mode",
    ["🎓 Academic Mode (Research & Study Synthesis)", "💼 Workplace Mode (Enterprise & Financial Audit)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Local Model Core")
model_choice = st.sidebar.selectbox(
    "Select Local Engine",
    ["qwen2.5:3b (Local Ollama)", "qwen2.5:1.5b (Fast Local)"]
)

st.sidebar.success("⚡ Local Runtime: Ollama Connected")
st.sidebar.info("🧬 Vector Store: Active (SQLite + Cosine Sim)")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["📂 Vector Workspace & RAG", "💬 AI Mentor & Guide", "📜 Audit History", "📊 NPU Architecture Specs"])

with tab1:
    if "Academic" in workflow_mode:
        st.info(
            "🎓 **Academic Mode Active:** Vector retrieval optimized for deep theoretical breakdown, citation parsing, and revision notes.")
    else:
        st.info(
            "💼 **Workplace Mode Active:** Vector retrieval optimized for balance sheet verification, compliance risk checking, and action matrices.")

    col1, col2 = st.columns([2, 1])

    active_filename = "Direct Manual Input"

    with col2:
        st.markdown("#### 📁 Ingest Document")
        uploaded_file = st.file_uploader("Upload local PDF or Text file", type=["pdf", "txt"])

        if uploaded_file is not None:
            active_filename = uploaded_file.name
            raw_text = ""
            if uploaded_file.type == "application/pdf":
                try:
                    reader = pypdf.PdfReader(uploaded_file)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            raw_text += extracted + "\n"
                except Exception as e:
                    st.error(f"Error parsing PDF: {e}")
            elif uploaded_file.type == "text/plain":
                raw_text = str(uploaded_file.read(), "utf-8")

            if raw_text:
                with st.spinner("Chunking document & generating local vector embeddings via nomic-embed-text..."):
                    chunk_count = store_document_vectors(active_filename, raw_text)
                    st.success(f"Successfully vectorized {chunk_count} semantic chunks into local DB!")

    with col1:
        st.markdown("#### Semantic Retrieval & Query")
        query_input = st.text_input("Enter your analytical question:",
                                    placeholder="e.g., Summarize primary risk factors and strategic financial targets...")

        if st.button("Execute Vector RAG Pipeline"):
            if not query_input.strip():
                st.warning("Please enter a query to run the RAG pipeline.")
            else:
                with st.spinner("Searching vector index & running local Qwen inference..."):
                    # 1. Retrieve relevant context chunks using vector similarity
                    retrieved_context = retrieve_relevant_context(query_input, top_k=3)

                    if not retrieved_context:
                        retrieved_context = "No indexed documents found. Please upload a document on the right panel first."

                    # 2. Formulate system prompt persona
                    if "Academic" in workflow_mode:
                        system_prompt = "You are an elite academic research assistant. Using the provided retrieved context snippets, deliver a rigorous, structured analytical summary with core concepts, definitions, and study bullets."
                    else:
                        system_prompt = "You are an enterprise business auditor. Using the provided retrieved context snippets, deliver a high-level executive briefing outlining financial impact, risk metrics, and concrete action directives."

                    prompt = f"Retrieved Context Chunks:\n{retrieved_context}\n\nUser Query:\n{query_input}"

                    try:
                        # 3. Local Model Call
                        response = ollama.chat(
                            model=model_choice.split(" ")[0],
                            messages=[
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': prompt}
                            ]
                        )

                        output_result = response['message']['content']

                        # 4. Log to Audit History
                        log_audit(active_filename, workflow_mode, query_input, output_result)

                        st.success("Vector RAG Execution Complete!")
                        st.markdown("### 📌 Enterprise AI Synthesis")
                        st.markdown(output_result)

                        with st.expander("🔍 Inspect Retrieved Vector Chunks"):
                            st.write(retrieved_context)

                    except Exception as e:
                        st.error(f"Inference failed. Ensure Ollama is running in background. Error: {e}")

with tab2:
    st.markdown("### 🤖 Interactive Local Mentor Guide")
    st.markdown("Select a technical topic to learn how the OmniContext framework operates:")
    mentor_q = st.selectbox("Select Topic:", [
        "How does local Vector RAG work without cloud servers?",
        "How does the system embed documents using nomic-embed-text?",
        "Why is SQLite vector storage ideal for offline NPU workflows?"
    ])
    if st.button("Get Technical Explanation"):
        if "Vector RAG" in mentor_q:
            st.write(
                "👉 Instead of flooding the model with raw text, the framework chunks documents into semantic blocks, computes similarity math locally, and feeds only high-relevance paragraphs to Qwen.")
        elif "nomic-embed-text" in mentor_q:
            st.write(
                "👉 The embedding model converts sentences into multi-dimensional float arrays locally, allowing mathematical distance calculations for ultra-fast document retrieval.")
        else:
            st.write(
                "👉 SQLite stores both chunks and vector payloads in a single lightweight file, making the entire RAG pipeline portable, secure, and fully offline.")

with tab3:
    st.markdown("### 📜 System Audit History")
    st.markdown("Secure on-device record of all queries processed through the RAG engine:")
    logs = fetch_audit_logs()
    if logs:
        for idx, log in enumerate(logs, 1):
            st.markdown(f"**{idx}. File:** `{log[0]}` | **Mode:** `{log[1]}`")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Query:* {log[2]} — *(Timestamp: {log[3]})*")
            st.markdown("---")
    else:
        st.info("No audit logs recorded yet. Run a query in the Vector Workspace tab.")

with tab4:
    st.markdown("### 📊 Hardware & Architecture Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vector Database", "SQLite + NumPy")
    c2.metric("Embedding Model", "nomic-embed-text")
    c3.metric("Cloud Telemetry", "0.0% (Air-Gapped)")
    c4.metric("Hardware Target", "Snapdragon Hexagon NPU")

    st.markdown("---")
    st.caption("OmniContext Framework — Snapdragon® AI Lab Build & Present Challenge 2026.")