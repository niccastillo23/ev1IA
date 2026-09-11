"""RAG pipeline for the fleet operations assistant.

Initializes a ChromaDB vector store from the operations manual
and returns a configured retriever for semantic search.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    LLM_API_KEY,
    LLM_EMBEDDING_MODEL,
    MANUAL_PATH,
)


def build_retriever():
    """Load, split, embed the operations manual and return a retriever.

    Returns:
        A LangChain retriever configured with k=3 for semantic search.
    """
    loader = TextLoader(str(MANUAL_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings(
        model=LLM_EMBEDDING_MODEL,
        api_key=LLM_API_KEY,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever
