import sys
import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from llm_rag import LLMRAGHandler
from conversation import ConversationManager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FAISS_INDEX_PATH = Path("faiss_index")

def process_new_pdfs(uploaded_files):
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue

        save_path = UPLOAD_DIR / file.name
        with save_path.open("wb") as f:
            f.write(file.read())

        st.sidebar.success(f"{file.name} saved.")

        try:
            with st.spinner(f"{file.name} is processed..."):
                st.session_state.llm.add_pdf_to_context(save_path)
        except Exception as exc:
            st.sidebar.error(f"Failed to index {file.name}: {exc}")
            continue

        st.session_state.processed_files.add(file.name)
        st.sidebar.success(f"{file.name} was indexed.")
        st.rerun()

conversation_manager = ConversationManager()
st.set_page_config(page_title="RAG Chatbot")
st.title("🤖 Chat with your PDFs 📄")

# Folder for saving uploaded PDFs
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

#########################
######### Initialization

# Initialize session state
if "llm" not in st.session_state:
    st.session_state.llm = LLMRAGHandler()
    saved_conversation = conversation_manager.load()
    if saved_conversation:
        st.session_state.llm.history = saved_conversation

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {p.name for p in UPLOAD_DIR.glob("*.pdf")}

def _indexed_pdf_names(vector_store) -> set[str]:
    names = set()
    for doc in vector_store.vector_store.docstore._dict.values():
        source = doc.metadata.get("source", "")
        if source:
            names.add(Path(source).name)
    return names


if "pdfs_synced" not in st.session_state:
    indexed_names = _indexed_pdf_names(st.session_state.llm.vector_store)
    for pdf_path in UPLOAD_DIR.glob("*.pdf"):
        if pdf_path.name in indexed_names:
            continue
        try:
            with st.spinner(f"Indexing {pdf_path.name}..."):
                st.session_state.llm.add_pdf_to_context(pdf_path)
            st.sidebar.success(f"Indexed {pdf_path.name}")
        except Exception as exc:
            st.sidebar.error(f"Could not index {pdf_path.name}: {exc}")
    st.session_state.pdfs_synced = True

if st.sidebar.button("Re-index all PDFs"):
    for pdf_path in UPLOAD_DIR.glob("*.pdf"):
        try:
            with st.spinner(f"Re-indexing {pdf_path.name}..."):
                st.session_state.llm.add_pdf_to_context(pdf_path)
        except Exception as exc:
            st.sidebar.error(f"Failed: {pdf_path.name}: {exc}")
    st.sidebar.success("Re-index complete.")
    st.rerun()


st.sidebar.subheader("📁 Already Saved PDFs:")
if st.session_state.processed_files:
    for pdf_path in st.session_state.processed_files :
        st.sidebar.markdown(f"- {pdf_path}")
else:
    st.sidebar.info("No PDFs uploaded yet.")
#########################




#########################
###### PDF Upload
st.sidebar.header("📄 Upload PDFs")

# Hochladen der PDF-Dateien
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Speichern & Anzeigen der Dateinamen
# TODO: These files should be directly loaded from vector store!
# TODO: Hier werden alle hochgeladenen files verarbeitet

if uploaded_files:
    process_new_pdfs(uploaded_files)

st.sidebar.header("🌐 Add Website URLs")
urls = st.sidebar.text_area("Enter website URLs (one per line)").splitlines()
if st.sidebar.button("📥 Add websites"):
    cleaned_urls = [u.strip() for u in urls if u and u.strip()]
    if not cleaned_urls:
        st.sidebar.warning("Enter at least one valid URL.")
    else:
        try:
            with st.spinner("Processing websites..."):
                added = st.session_state.llm.vector_store.index_websites(cleaned_urls)
            if added:
                st.sidebar.success(f"Indexed {len(added)} document chunk(s).")
            else:
                st.sidebar.warning("No content found at those URLs.")
        except Exception as exc:
            st.sidebar.error(f"Website indexing failed: {exc}")
#########################


#####################
##### Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    try:
        with st.spinner("Thinking..."):
            st.session_state.llm.generate_response(user_input)
    except Exception as exc:
        st.error(f"Could not generate a response: {exc}")
    else:
        conversation_manager.save(st.session_state.llm.get_history())
        st.rerun()

# Display chat messages
for msg in st.session_state.llm.get_history():
    if isinstance(msg, SystemMessage):
        # do not display instructions for the chatbot
        continue
    
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
#########################


#########################
# Sidebar actions
if st.sidebar.button("🗑️ Reset Conversation"):
    st.session_state.llm.reset()
    conversation_manager.clear()
    st.rerun()