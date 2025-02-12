from logging import getLogger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph_project.settings import settings


index_logger = getLogger("index")

def create_index(documents: list[Document]) -> Chroma:
    """
    Create a simple index using the in memory Chroma
    """
    index_logger.info(f"Processing Index for {len(documents)} docs")
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=100, chunk_overlap=50
    # )
    # doc_splits = text_splitter.split_documents(documents)
    index = Chroma.from_documents(
        documents, 
        OpenAIEmbeddings(
            model=settings.embeddings_model_name, 
            api_key=settings.openai_api_key.get_secret_value()
        ),
    )
    return index
