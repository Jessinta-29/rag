from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatOpenAI(
    model_name="meta-llama/Llama-3-70b-chat-hf",
    openai_api_key="tgp_v1_18FOkOVd_p9hfguK0kEUcF362Nw61eWBaQto-hPA3hg",
    openai_api_base="https://api.together.xyz/v1",
    temperature=0.3
)