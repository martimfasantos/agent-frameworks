import os
from logging import getLogger
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader


loader_logger = getLogger("loader")

def load_documents_from_folder(knowledge_base_path: str):
    """
    Load all `.md` files from knowledge_base_path folder and return them as a list of Document.
    Args:
        knowledge_base_path (str): Path to the knowledge base containing `.md` files.
    Returns:
        List[Document]: A list of Document objects.
    """
    print(f"Loading documents from {knowledge_base_path}")
    loader = DirectoryLoader(knowledge_base_path, glob="*.md", show_progress=True)
    docs = loader.load()
    loader_logger.info(f"Read {len(docs)} files")

    return docs