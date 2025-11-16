# app.py
import streamlit as st
import os
import tempfile
import subprocess
import cv2
import speech_recognition as sr
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from uuid import uuid4
from datetime import datetime
import json
import shutil
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))


# Optional KB dependencies (used if available)
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

# PDF reader
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

load_dotenv()

# Configure your LLM / generative model (existing)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
search_tool = DuckDuckGoSearchRun()

st.set_page_config(page_title="OmniSight AI", layout="wide")


# --- ffmpeg availability check ---
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    # show an informational warning in the UI when the app runs
    try:
        st.warning("ffmpeg not found on system PATH. Video audio extraction/transcription will be unavailable. Install ffmpeg and restart the app.")
    except Exception:
        pass

# ---------- Persistence config ----------
SAVED_DIR = "saved_analyses"
FILES_DIR = os.path.join(SAVED_DIR, "files")
ANALYSES_DB = os.path.join(SAVED_DIR, "analyses.json")

KB_DIR = "kb"
KB_RAW_DIR = os.path.join(KB_DIR, "raw")
KB_CHUNKS_FILE = os.path.join(KB_DIR, "chunks.jsonl")  # newline-delimited chunks meta/text
KB_CHROMA_DIR = os.path.join(KB_DIR, "chroma_db")  # chroma persistence dir

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(KB_RAW_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)

if not os.path.exists(ANALYSES_DB):
    with open(ANALYSES_DB, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


# -----------------------
# Helper functions (persistence + file helpers)
# -----------------------
def load_analyses():
    try:
        with open(ANALYSES_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_analyses_list(lst):
    with open(ANALYSES_DB, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=2, ensure_ascii=False)


def append_analysis(entry):
    lst = load_analyses()
    lst.append(entry)
    save_analyses_list(lst)


def delete_analysis_by_id(entry_id):
    lst = load_analyses()
    new_lst = [e for e in lst if e["id"] != entry_id]
    removed = [e for e in lst if e["id"] == entry_id]
    for e in removed:
        for key in ("saved_file", "thumbnail"):
            p = e.get(key)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    save_analyses_list(new_lst)


def save_temp_file_from_uploaded(uploaded_file, dest_dir=FILES_DIR, forced_name=None):
    if uploaded_file is None:
        return None
    original_name = uploaded_file.name
    suffix = f".{original_name.split('.')[-1]}" if "." in original_name else ""
    if forced_name:
        fname = forced_name
    else:
        fname = f"{uuid4().hex}_{original_name}"
    save_path = os.path.join(dest_dir, fname)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path



def extract_thumbnail(video_path, max_tries=5):
    """
    Robust thumbnail extraction:
    - Try cv2.VideoCapture and read up to max_tries frames.
    - If that fails, fallback to ffmpeg (if available).
    Returns thumbnail_path or None.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return _ffmpeg_extract_frame_fallback(video_path)
        frame = None
        for i in range(max_tries):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
        try:
            cap.release()
        except Exception:
            pass
        if frame is None:
            return _ffmpeg_extract_frame_fallback(video_path)
        thumb_path = f"{video_path}_thumb.jpg"
        cv2.imwrite(thumb_path, frame)
        return thumb_path
    except Exception:
        return _ffmpeg_extract_frame_fallback(video_path)

def _ffmpeg_extract_frame_fallback(video_path):
    """
    Use ffmpeg to extract a single frame as a thumbnail. Requires ffmpeg on PATH.
    """
    if shutil.which("ffmpeg") is None:
        return None
    thumb_path = f"{video_path}_thumb.jpg"
    cmd = ["ffmpeg", "-i", video_path, "-vf", "scale=640:-1", "-frames:v", "1", "-y", thumb_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode == 0 and os.path.exists(thumb_path):
            return thumb_path
    except Exception:
        pass
    return None



def transcribe_audio(video_path, audio_timeout=60):
    """
    Extract audio using ffmpeg and transcribe via SpeechRecognition (Google).
    Returns transcription string (or helpful error message).
    """
    if shutil.which("ffmpeg") is None:
        return "Transcription unavailable: ffmpeg not installed or not on PATH."

    audio_path = video_path + "_audio.wav"
    try:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=audio_timeout)
        if proc.returncode != 0 or not os.path.exists(audio_path):
            stderr = proc.stderr.decode(errors="ignore") if proc.stderr else ""
            return f"Transcription failed: ffmpeg error. {stderr[:300]}"
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                transcription = "Transcription empty: audio not understandable."
            except sr.RequestError as e:
                transcription = f"Transcription service error: {e}"
    except subprocess.TimeoutExpired:
        transcription = "Transcription failed: ffmpeg timed out extracting audio."
    except Exception as e:
        transcription = f"Transcription failed: {e}"
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
    return transcription


# -----------------------
# LLM analysis wrapper (keeps existing behavior)
# -----------------------
def analyze_content(prompt, image_path=None, transcription=None):
    content = prompt
    if transcription:
        content += f"\n\nTranscription:\n{transcription}"

    try:
        if image_path:
            with open(image_path, "rb") as img:
                response = model.generate_content([content, {"mime_type": "image/jpeg", "data": img.read()}])
        else:
            response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Analysis failed: {e}"


def perform_web_search(query):
    results = search_tool.run(query)
    prompt = f"Web search results for '{query}':\n{results}\n\nProvide a comprehensive analysis."
    return analyze_content(prompt)


# -----------------------
# KB utilities: chunking, ingestion, embeddings, query
# -----------------------
# Simple text splitter (character-based)
def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# Load PDF/txt/md
def load_document_text(path):
    ext = path.split('.')[-1].lower()
    if ext in ("txt", "md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == "pdf" and PDF_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(path)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
        except Exception:
            return ""
    # unsupported or PyPDF2 not available
    return ""


# KB storage of chunks (fallback store when chroma not available)
def append_chunks_to_jsonl(chunks_with_meta):
    # each item: {"id": id, "source": source, "text": text, "offset": offset}
    with open(KB_CHUNKS_FILE, "a", encoding="utf-8") as f:
        for c in chunks_with_meta:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_all_chunks_from_jsonl():
    if not os.path.exists(KB_CHUNKS_FILE):
        return []
    out = []
    with open(KB_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line.strip()))
            except Exception:
                pass
    return out


# Embeddings: if sentence-transformers available, use it; else None
EMBEDDER = None
if CHROMADB_AVAILABLE:
    try:
        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        EMBEDDER = None

# Initialize chroma client & collection if possible
CHROMA_CLIENT = None
CHROMA_COLLECTION = None
if CHROMADB_AVAILABLE:
    try:
        CHROMA_CLIENT = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=KB_CHROMA_DIR))
        # create or get collection
        COLLECTION_NAME = "kb_collection"
        try:
            CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(COLLECTION_NAME)
        except Exception:
            CHROMA_COLLECTION = CHROMA_CLIENT.create_collection(COLLECTION_NAME)
    except Exception:
        CHROMA_CLIENT = None
        CHROMA_COLLECTION = None


def ingest_file_to_kb(file_path, doc_name=None, chunk_size=1000, overlap=200):
    """
    Ingest a single file into KB:
    - extract text
    - chunk
    - if chroma + embedder available: compute embeddings and upsert to chroma
    - else: append chunks to local jsonl
    """
    # ensure we're modifying module-level chroma vars if needed
    global CHROMA_CLIENT, CHROMA_COLLECTION

    if doc_name is None:
        doc_name = os.path.basename(file_path)
    text = load_document_text(file_path)
    if not text:
        return {"status": "empty", "msg": "No extracted text (unsupported file or empty)"}
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    chunks_meta = []
    for i, c in enumerate(chunks):
        chunks_meta.append({
            "id": f"{uuid4().hex}",
            "source": doc_name,
            "text": c,
            "offset": i
        })

    if CHROMA_COLLECTION is not None and EMBEDDER is not None:
        # compute embeddings and upsert
        try:
            texts = [c["text"] for c in chunks_meta]
            embeddings = EMBEDDER.encode(texts, show_progress_bar=False).tolist()
            ids = [c["id"] for c in chunks_meta]
            metadatas = [{"source": c["source"], "offset": c["offset"]} for c in chunks_meta]
            CHROMA_COLLECTION.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            CHROMA_CLIENT.persist()
            return {"status": "ok", "method": "chroma", "added": len(chunks_meta)}
        except Exception as e:
            # fallback to jsonl
            append_chunks_to_jsonl(chunks_meta)
            return {"status": "ok_fallback", "msg": str(e)}
    else:
        append_chunks_to_jsonl(chunks_meta)
        return {"status": "ok_fallback", "added": len(chunks_meta)}


def rebuild_kb_index(chunk_size=1000, overlap=200):
    """
    Rebuild KB from raw files in KB_RAW_DIR. Clears chroma (if present) and jsonl.
    """
    # ensure we can modify module-level chroma vars
    global CHROMA_CLIENT, CHROMA_COLLECTION

    # clear jsonl
    if os.path.exists(KB_CHUNKS_FILE):
        os.remove(KB_CHUNKS_FILE)
    # reset chroma if available
    if CHROMA_COLLECTION is not None:
        try:
            # delete the collection by name and recreate a fresh one
            try:
                # If chroma client exposes delete_collection by name
                CHROMA_CLIENT.delete_collection(CHROMA_COLLECTION.name)
            except Exception:
                # fallback: attempt to delete by collection name variable if available
                try:
                    CHROMA_CLIENT.delete_collection("kb_collection")
                except Exception:
                    pass
            CHROMA_COLLECTION = CHROMA_CLIENT.create_collection("kb_collection")
        except Exception:
            # swallow errors (we'll fallback to jsonl chunks)
            pass

    files = sorted([os.path.join(KB_RAW_DIR, f) for f in os.listdir(KB_RAW_DIR)], key=os.path.getmtime)
    added_total = 0
    results = []
    for fp in files:
        res = ingest_file_to_kb(fp, doc_name=os.path.basename(fp), chunk_size=chunk_size, overlap=overlap)
        results.append((fp, res))
        if res.get("added"):
            added_total += res["added"]
    return {"files": len(files), "added_chunks": added_total, "results": results}


def query_kb(query, top_k=5):
    """
    Query KB:
    - If chroma+embedder available: use vector search
    - Else: simple substring/keyword search across jsonl chunks
    Returns list of dicts: {"source","text","offset","score"}
    """
    if CHROMA_COLLECTION is not None and EMBEDDER is not None:
        try:
            q_emb = EMBEDDER.encode([query])[0].tolist()
            res = CHROMA_COLLECTION.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
            docs = []
            if res and "documents" in res and len(res["documents"]) > 0:
                docs_list = res["documents"][0]
                metas_list = res["metadatas"][0]
                dists_list = res["distances"][0]
                for text, meta, dist in zip(docs_list, metas_list, dists_list):
                    docs.append({"source": meta.get("source"), "text": text, "offset": meta.get("offset"), "score": float(dist)})
            return docs
        except Exception as e:
            # fallback to simple search
            pass

    # fallback: keyword/substring search
    all_chunks = load_all_chunks_from_jsonl()
    hits = []
    qlow = query.lower()
    for c in all_chunks:
        if qlow in c.get("text", "").lower() or any(tok in c.get("text", "").lower() for tok in qlow.split()):
            hits.append({"source": c.get("source"), "text": c.get("text"), "offset": c.get("offset"), "score": 1.0})
            if len(hits) >= top_k:
                break
    return hits


# RAG answer builder (bounded prompt)
def rag_answer_from_chunks(query, chunks, max_context_chars=3000):
    """
    Build a prompt with top chunks and ask the LLM to answer using them.
    """
    if not chunks:
        return "No relevant KB passages found."

    included = []
    total = 0
    for c in chunks:
        t = c.get("text", "")
        if not t:
            continue
        if total + len(t) > max_context_chars:
            # include a prefix if it doesn't fit entirely
            remaining = max_context_chars - total
            if remaining > 50:
                included.append(f"[{c.get('source')}] {t[:remaining]}...")
                total += remaining
            break
        included.append(f"[{c.get('source')}] {t}")
        total += len(t)

    prompt = (
        "You are a helpful assistant. Use the following extracted passages from the knowledge base to answer the user's question.\n\n"
        "Passages:\n" + "\n\n".join(included) + "\n\n"
        f"User question: {query}\n\n"
        "Answer the question using only the information in the passages. If the answer is not present, say you don't know. When you refer to a passage, cite the source in square brackets (e.g., [filename.pdf])."
    )
    # use analyze_content (which calls the configured LLM)
    return analyze_content(prompt)


# -----------------------
# Sidebar (navigation only)
# -----------------------
with st.sidebar:
    st.markdown("### Navigate")
    # provide a non-empty label and hide it visually to avoid the accessibility warning
    nav_choice = st.radio("Choose page", ["New Analysis", "Past Analyses", "Knowledge Base Search"], label_visibility="collapsed")

    st.markdown("---")
    st.caption("Use the main area to pick analysis type and upload files.")


# -----------------------
# Main area header
# -----------------------
st.markdown("<h1 style='margin-bottom:0.25rem'>OmniSight AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;margin-top:0'>Advanced multimedia analysis with semantic vector storage and knowledge base</p>", unsafe_allow_html=True)
st.write("")  # small spacer

# -----------------------
# Past Analyses page
# -----------------------
if nav_choice == "Past Analyses":
    st.subheader("Past Analyses")
    analyses = load_analyses()
    if not analyses:
        st.info("No past analyses saved yet.")
    else:
        analyses_sorted = sorted(analyses, key=lambda x: x.get("timestamp", ""), reverse=True)
        labels = [
            f"{datetime.fromisoformat(a['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} — {a['type']} — {a.get('original_name','n/a')}"
            for a in analyses_sorted
        ]
        idx = st.selectbox("Select an analysis to view", options=list(range(len(labels))), format_func=lambda i: labels[i])
        selected = analyses_sorted[idx]

        st.markdown("### Details")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write(f"**Type:** {selected.get('type')}")
            st.write(f"**Original file name:** {selected.get('original_name', 'N/A')}")
            st.write(f"**Saved file:** {os.path.basename(selected.get('saved_file','')) if selected.get('saved_file') else 'N/A'}")
            st.write(f"**Timestamp:** {selected.get('timestamp')}")
            if selected.get("type") == "Image":
                if selected.get("saved_file") and os.path.exists(selected["saved_file"]):
                    st.image(selected["saved_file"], caption="Saved image", use_column_width=True)
                else:
                    st.warning("Saved image file not found.")
            elif selected.get("type") == "Video":
                if selected.get("saved_file") and os.path.exists(selected["saved_file"]):
                    st.video(selected["saved_file"])
                else:
                    st.warning("Saved video file not found.")
            elif selected.get("type") == "Web Search":
                st.write("Web search analysis (no file).")
        with col2:
            st.markdown("### Analysis output")
            st.write(selected.get("analysis_text", "No analysis text saved."))
            if selected.get("transcription"):
                st.markdown("### Transcription")
                st.write(selected.get("transcription"))

            st.markdown("### Actions")
            download_payload = json.dumps(selected, indent=2, ensure_ascii=False)
            st.download_button("Download analysis (JSON)", download_payload.encode("utf-8"), file_name=f"analysis_{selected['id']}.json")
            if selected.get("saved_file") and os.path.exists(selected["saved_file"]):
                with open(selected["saved_file"], "rb") as f:
                    b = f.read()
                st.download_button("Download original file", b, file_name=os.path.basename(selected["saved_file"]))

            if st.button("Delete this analysis"):
                delete_analysis_by_id(selected["id"])
                st.success("Deleted. Refreshing list...")
                st.experimental_rerun()

    st.markdown("---")
    st.markdown("## Instructions")
    st.markdown(
        """
- This page lists previously saved analyses.
- Select any item to view details, download the original file or the analysis JSON.
- Delete removes the saved files and the record.
"""
    )
    st.stop()


# -----------------------
# Knowledge Base Search page
# -----------------------
if nav_choice == "Knowledge Base Search":
    st.subheader("Knowledge Base Search")

    # Show availability
    if CHROMADB_AVAILABLE and EMBEDDER is not None and CHROMA_CLIENT is not None:
        st.success("KB features: Chroma + sentence-transformers are available (high-quality semantic search).")
    else:
        st.warning("Chroma or sentence-transformers not available — KB will use a simple keyword fallback. For best results `pip install chromadb sentence-transformers PyPDF2`.")

    st.markdown("### Upload documents to KB")
    uploaded = st.file_uploader("Upload PDF/TXT/MD files (you can upload multiple)", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            # save to KB_RAW_DIR
            dest = os.path.join(KB_RAW_DIR, f"{uuid4().hex}_{f.name}")
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            st.write(f"Saved: {f.name} → {dest}")

    if st.button("Rebuild KB index (ingest all uploaded docs)"):
        with st.spinner("Ingesting documents..."):
            res = rebuild_kb_index()
        st.success(f"Ingestion complete. Files processed: {res['files']}, chunks added: {res['added_chunks']}")
        st.write("Details:")
        for fp, r in res["results"]:
            st.write(f"- {os.path.basename(fp)}: {r}")

    st.markdown("---")
    st.markdown("### KB Query")
    kb_query = st.text_input("Enter knowledge-base search query:")
    top_k = st.slider("Top-k passages to retrieve", 1, 10, 4)
    if st.button("Search KB"):
        if not kb_query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Querying KB..."):
                hits = query_kb(kb_query, top_k=top_k)
            if not hits:
                st.info("No relevant passages found.")
            else:
                st.markdown("#### Top passages")
                for i, h in enumerate(hits):
                    st.markdown(f"**{i+1}. Source:** {h.get('source')}  —  **score:** {h.get('score')}")
                    st.write(h.get("text")[:300] + ("..." if len(h.get("text",""))>300 else ""))

                if st.button("Generate Answer with RAG"):
                    with st.spinner("Generating answer using the retrieved passages..."):
                        answer = rag_answer_from_chunks(kb_query, hits)
                    st.markdown("### RAG Answer")
                    st.write(answer)

    st.markdown("---")
    st.markdown("### Current KB files")
    files = sorted([f for f in os.listdir(KB_RAW_DIR)], key=lambda x: os.path.getmtime(os.path.join(KB_RAW_DIR, x)), reverse=True)
    if files:
        for fn in files:
            fnp = os.path.join(KB_RAW_DIR, fn)
            st.write(f"- {fn} ({os.path.getsize(fnp)} bytes)")
            if st.button(f"Delete {fn}"):
                try:
                    os.remove(fnp)
                    st.success("Deleted. Rebuild index to remove its chunks.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
    else:
        st.info("No KB files uploaded yet. Upload files above and click 'Rebuild KB index' to ingest.")

    st.markdown("---")
    st.markdown(
        """
KB notes:
- In the best case, install `chromadb`, `sentence-transformers` and `PyPDF2` to get true semantic retrieval and PDF ingestion.
- Chunks are stored in `kb/chunks.jsonl` (fallback) and Chromadb data is persisted in `kb/chroma_db/`.
- Rebuild the index whenever you add/remove files.
"""
    )
    st.stop()


# -----------------------
# New Analysis (main area)
# -----------------------
analysis_type = st.selectbox("Choose analysis type", ["Image", "Video", "Web Search"])

with st.container():
    if analysis_type == "Image":
        st.markdown("#### Upload an image")

        image_file = st.file_uploader(
            "Drag & drop image here",
            type=["jpg", "png", "jpeg"],
            key="img_uploader"
        )

        analyze = st.button("Analyze Image")

        if analyze:
            if image_file is None:
                st.warning("Please upload an image before analyzing.")
            else:
                saved_path = save_temp_file_from_uploaded(image_file)
                prompt = "Provide a detailed analysis of this image."
                with st.spinner("Analyzing image..."):
                    result_text = analyze_content(prompt, image_path=saved_path)
                st.image(saved_path, width=450, caption="Uploaded image")
                st.markdown("### Analysis Result")
                st.markdown(result_text)

                entry = {
                    "id": uuid4().hex,
                    "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Image",
                    "original_name": image_file.name,
                    "saved_file": saved_path,
                    "analysis_text": result_text
                }
                append_analysis(entry)
                st.success("Analysis saved to Past Analyses.")

    elif analysis_type == "Video":
        st.markdown("#### Upload a video")

        video_file = st.file_uploader(
            "Drag & drop video here",
            type=["mp4", "avi", "mov"],
            key="vid_uploader"
        )

        analyze = st.button("Analyze Video")

        if analyze:
            if video_file is None:
                st.warning("Please upload a video before analyzing.")
            else:
                saved_path = save_temp_file_from_uploaded(video_file)
                thumbnail_path = None
                transcription = ""
                try:
                    with st.spinner("Extracting thumbnail and transcribing audio..."):
                        thumbnail_path = extract_thumbnail(saved_path)
                        transcription = transcribe_audio(saved_path)

                    if thumbnail_path and os.path.exists(thumbnail_path):
                        st.image(thumbnail_path, caption="Video Thumbnail", width=420)

                    st.markdown("### Transcription")
                    st.write(transcription)

                    prompt = "Analyze this video based on the thumbnail and transcription."
                    with st.spinner("Analyzing video..."):
                        result_text = analyze_content(prompt, image_path=thumbnail_path, transcription=transcription)

                    st.markdown("### Analysis Result")
                    st.markdown(result_text)

                    entry = {
                        "id": uuid4().hex,
                        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Video",
                        "original_name": video_file.name,
                        "saved_file": saved_path,
                        "thumbnail": thumbnail_path,
                        "transcription": transcription,
                        "analysis_text": result_text
                    }
                    append_analysis(entry)
                    st.success("Analysis saved to Past Analyses.")

                except Exception as e:
                    st.error(f"Error during video processing: {e}")

    else:  # Web Search
        st.markdown("#### Web Search & Analyze")

        query = st.text_input("Enter search query:")

        if st.button("Search & Analyze"):
            if query == "":
                st.warning("Please enter a query before running the search.")
            else:
                with st.spinner("Running web search and generating analysis..."):
                    result_text = perform_web_search(query)

                st.markdown("### Analysis Result")
                st.markdown(result_text)

                entry = {
                    "id": uuid4().hex,
                    "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Web Search",
                    "original_name": f"websearch_{query[:40]}",
                    "saved_file": None,
                    "analysis_text": result_text,
                    "query": query
                }
                append_analysis(entry)
                st.success("Analysis saved to Past Analyses.")

st.markdown("---")
left, right = st.columns([2, 5])
with left:
    st.markdown("### Instructions")
    st.markdown(
        """
- Select analysis type from the dropdown.
- Provide required input (upload files or enter queries).
- Click analyze to generate and save analyses.
"""
    )




# import streamlit as st
# import os
# import tempfile
# import subprocess
# import cv2
# import speech_recognition as sr
# import google.generativeai as genai
# from langchain_community.tools import DuckDuckGoSearchRun
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = genai.GenerativeModel('')
# search_tool = DuckDuckGoSearchRun()

# st.set_page_config(page_title="AI Multimodal Analyzer", layout="wide")
# st.title("AI Multimodal Analyzer")
# st.markdown("Analyze media or perform web searches using Gemini AI")

# def save_temp_file(uploaded_file):
#     if uploaded_file:
#         suffix = f".{uploaded_file.name.split('.')[-1]}"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(uploaded_file.getbuffer())
#             return tmp.name
#     return None

# def extract_thumbnail(video_path):
#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     cap.release()
#     if ret:
#         thumb_path = video_path + "_thumb.jpg"
#         cv2.imwrite(thumb_path, frame)
#         return thumb_path
#     return None

# def transcribe_audio(video_path):
#     audio_path = video_path + "_audio.wav"
#     subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],
#                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     recognizer = sr.Recognizer()
#     transcription = ""
#     with sr.AudioFile(audio_path) as source:
#         audio = recognizer.record(source)
#         try:
#             transcription = recognizer.recognize_google(audio)
#         except Exception as e:
#             transcription = f"Transcription failed: {e}"
#     os.remove(audio_path)
#     return transcription

# def analyze_content(prompt, image_path=None, transcription=None):
#     content = prompt
#     if transcription:
#         content += f"\n\nTranscription:\n{transcription}"

#     try:
#         if image_path:
#             with open(image_path, "rb") as img:
#                 response = model.generate_content([content, {"mime_type": "image/jpeg", "data": img.read()}])
#         else:
#             response = model.generate_content(content)
#         return response.text
#     except Exception as e:
#         return f"Analysis failed: {e}"
    
# def perform_web_search(query):
#     results = search_tool.run(query)
#     prompt = f"Web search results for '{query}':\n{results}\n\nProvide a comprehensive analysis."
#     return analyze_content(prompt)

# analysis_type = st.sidebar.radio("Choose analysis type:", ["Image", "Video", "Web Search"])

# if analysis_type == "Image":
#     image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
#     if st.button("Analyze Image") and image_file:
#         image_path = save_temp_file(image_file)
#         st.image(image_path, width=400)
#         prompt = "Provide a detailed analysis of this image."
#         result = analyze_content(prompt, image_path=image_path)
#         st.markdown(result)
#         os.remove(image_path)
    
# elif analysis_type == "Video":
#     video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#     if st.button("Analyze Video") and video_file:
#         video_path = save_temp_file(video_file)
#         thumbnail_path = extract_thumbnail(video_path)
#         transcription = transcribe_audio(video_path)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(thumbnail_path, caption="Thumbnail", width=350)
#         with col2:
#             st.markdown("### Transcription")
#             st.write(transcription)

#         prompt = "Analyze this video based on the thumbnail and transcription."
#         result = analyze_content(prompt, image_path=thumbnail_path, transcription=transcription)
#         st.markdown(result)

#         os.remove(video_path)
#         os.remove(thumbnail_path)

# elif analysis_type == "Web Search":
#     query = st.text_input("Enter search query:")
#     if st.button("Search & Analyze") and query:
#         result = perform_web_search(query)
#         st.markdown(result)

# st.markdown("## Instructions")
# st.markdown(
#     """
# - Select analysis type from sidebar.
# - Provide required input (upload files or enter queries).
# - Click the analyze button to view results.
# """
# )




