from logging import getLogger
from llama_index.core import Document, VectorStoreIndex
from autogen_project.settings import settings


index_logger = getLogger("index")

def create_index(documents: list[Document]) -> VectorStoreIndex:
    """
    Create a simple index using the in memory VectorStoreIndex
    """
    index_logger.info(f"Processing Index for {len(documents)} docs")
    index = VectorStoreIndex.from_documents(
        documents, embed_model=f"local:{settings.local_embedding_model}"
    )
    return index
