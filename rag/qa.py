from langchain.chains import RetrievalQA
from rag.indexing import load_qdrant_index

def query_qdrant(query, embedding_model, llm):
    db = load_qdrant_index(embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    # Debugging
    relevant_docs = retriever.get_relevant_documents(query)
    print(f"🔍 Retrieved docs: {relevant_docs}")
    
    return qa_chain.invoke(query)
