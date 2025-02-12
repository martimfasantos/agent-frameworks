from langchain_core.tools import tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph_project.vector_store.loader import load_documents_from_folder
from langgraph_project.vector_store.index import create_index
from langgraph_project.settings import settings


# Initialize the VectorStore and set the index
docs = load_documents_from_folder(settings.knowledge_base_path)
index = create_index(docs)

@tool
def retrieve_information_vectorbase(query: str):
    """Retrieve information from a MongoDB Atlas Vector Search index.
    
    Args:
        query (str): The search query.

    Returns:
        str: Retrieved documents concatenated as a string.
    """
    documents = index.similarity_search(query=query, k=1)
    return "\n\n".join([doc.page_content for doc in documents]) 
