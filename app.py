import streamlit as st
from rag.youtube_loader import load_youtube_transcript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.file_loader import load_file
from rag.indexing import create_qdrant_index
from rag.qa import query_qdrant
from models.models import embedding_model, llm

st.set_page_config(page_title="Transcript Q&A", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcript_uploaded" not in st.session_state:
    st.session_state.transcript_uploaded = False
if "index_built" not in st.session_state:
    st.session_state.index_built = False

col1, col2 = st.columns([1, 2])

# ------------------------- LEFT PANEL -------------------------
with col1:
    st.header("Upload or Import Transcript")

    source_option = st.radio("Choose input source:", ["Upload File", "YouTube URL"])

    if source_option == "Upload File":
        uploaded_file = st.file_uploader("Upload a PDF, TXT, CSV, or DOCX", type=["pdf", "txt", "csv", "docx"])
        doc_type = st.selectbox("Select document type (optional)", ["Auto", "PDF", "TXT", "CSV", "DOCX"])

        if uploaded_file and not st.session_state.index_built:
            st.session_state.transcript_uploaded = False
            with st.spinner("Processing file..."):
                docs = load_file(uploaded_file)
                if docs:
                    try:
                        create_qdrant_index(docs, embedding_model)
                        st.session_state.transcript_uploaded = True
                        st.session_state.index_built = True
                        st.success("File uploaded and indexed successfully!")
                    except Exception as e:
                        st.warning(f"Index creation failed. {str(e)}")
                else:
                    st.error("Unsupported file format.")

    elif source_option == "YouTube URL":
        video_url = st.text_input("Enter YouTube Video URL")

        if st.button("Fetch YouTube Transcript"):
            if video_url:
                docs, error = load_youtube_transcript(video_url)
                if docs:
                    try:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)
                        create_qdrant_index(chunks, embedding_model)
                        st.session_state.transcript_uploaded = True
                        st.session_state.index_built = True
                        st.success("Transcript downloaded and indexed successfully from YouTube!")
                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
                else:
                    st.error(f"Failed to fetch transcript: {error}")
            else:
                st.warning("Please enter a valid YouTube URL.")

# ------------------------- RIGHT PANEL -------------------------
with col2:
    st.header("Query Dashboard")
    user_query = st.text_input("Ask anything about the uploaded transcript")

    run_query = st.button("Ask")

    if run_query and user_query:
        if not st.session_state.transcript_uploaded:
            st.warning("Please upload or fetch a transcript before asking questions.")
        else:
            try:
                with st.spinner("Searching..."):
                    answer = query_qdrant(user_query, embedding_model, llm)
                    result_text = answer["result"] if isinstance(answer, dict) and "result" in answer else str(answer)
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("Bot", result_text))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**You:** {msg}" if sender == "You" else f"**Bot:** {msg}")
