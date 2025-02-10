from langchain_core.tools import tool
from langchain_mongodb import MongoDBAtlasVectorSearch


@tool
async def retrieve_information_vectorbase(query: str, index: MongoDBAtlasVectorSearch):
    documents = index.asimilarity_search(query=query, k=2)
    return "\n\n".join([doc["page_content"] for doc in documents]) 
