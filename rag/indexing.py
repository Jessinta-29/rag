from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Your Qdrant configuration
QDRANT_URL = "https://8a2e3d09-4e5c-4a34-a733-880ce6a2720e.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.KjPDtZ0jy7E0e2fHUzbKscGfykT6DHvzcPeRpz3hBh0"
COLLECTION_NAME = "transcript_kb"

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def create_qdrant_index(docs, embedding_model):
    """Split, embed, and upload documents to Qdrant."""
    if not docs:
        print("[WARN] No documents to index.")
        return False
    try:
        print("📄 Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"✅ {len(chunks)} chunks created.")

        if not chunks:
            print("[ERROR] No chunks to index. Exiting.")
            return False

        dim = len(embedding_model.embed_query("test"))
        print(f"📐 Vector dimension: {dim}")

        print(f"🧹 Removing existing knowledge base: '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print("✅ Collection recreated. Old content deleted.")

        print("📤 Uploading documents to Qdrant...")
        Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(f"Qdrant collection now contains {count} documents.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create Qdrant index: {e}")
        return False

def load_qdrant_index(embedding_model):
    """Load Qdrant index for querying."""
    print("Loading Qdrant index for retrieval...")
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model
    )

def query_qdrant(query, embedding_model, llm):
    """Query Qdrant and only answer from indexed data."""
    db = load_qdrant_index(embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print(f"🔍 Query: {query}")
    docs = retriever.get_relevant_documents(query)
    print(f"🔍 Retrieved {len(docs)} documents")

    if not docs:
        return {"result": "No relevant information found in the uploaded file."}

    for i, doc in enumerate(docs[:3]):
        print(f"[{i+1}] {doc.page_content[:200]}...")

    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    return chain({"input_documents": docs, "question": query})
