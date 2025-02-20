import asyncio
from typing import List
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.selectors import LLMSingleSelector
from settings import settings


async def main():
    # load documents
    documents = SimpleDirectoryReader(input_files=["docs/metagpt.pdf"]).load_data()

    # split the documents into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # define the LLMs
    Settings.llm = OpenAI(
        model=settings.openai_model_name, 
        api_key=settings.openai_api_key.get_secret_value()
    )
    Settings.embed_model = OpenAIEmbedding(
        model=settings.embeddings_model_name,
        api_key=settings.openai_api_key.get_secret_value()
    )

    # ----------------------------------------
    #                 TOOLS
    # ----------------------------------------

    # Simple tool
    def add(x: int, y: int) -> int:        # type annotations and docstrings are used 
        """Adds two integers together."""  # as a prompt for the llm
        return x + y
    
    add_tool = FunctionTool.from_defaults(fn=add)

    # RAG tool
    vector_index = VectorStoreIndex(nodes)

    def vector_query(
        query: str, 
        page_numbers: List[str]     # is infered by the llm based on the prompt
    ) -> str:
        """Perform a vector search over an index.
        
        query (str): the string query to be embedded.
        page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
            over all pages. Otherwise, filter by the set of specified pages.
        
        """

        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
    
    vector_query_tool = FunctionTool.from_defaults(
        name="vector_tool",
        fn=vector_query
    )


    # Call the LLM
    # Using the RAG tool
    response = Settings.llm.predict_and_call(
        [add_tool, vector_query_tool], 
        "What are the high-level results of MetaGPT as described on page 2?", 
        verbose=True
    )
    print(str(response))
    """
    === Calling Function ===
    Calling function: vector_tool with args: {"query": "high-level results of MetaGPT", "page_numbers": ["2"]}
    === Function Output ===
    MetaGPT achieves a new state-of-the-art performance in code generation benchmarks, with Pass@1 scores of 85.9% and 87.7% on ...
    """
    
    for n in response.source_nodes:
        print(n.metadata) 
    """
    {'page_label': '2', 'file_name': 'metagpt.pdf', 'file_path': 'docs/metagpt.pdf', 'file_type': 'application/pdf', 
    'file_size': 16911937, 'creation_date': '2025-02-14', 'last_modified_date': '2025-02-14'}
    """
    print("-"*50)


    # Using the simple tool
    response = Settings.llm.predict_and_call(
        [add_tool, vector_query_tool], 
        "What is the result of adding 2 and 3?", 
        verbose=True
    )
    print(str(response)) 
    """
    === Calling Function ===
    Calling function: add with args: {"x": 2, "y": 3}
    === Function Output ===
    5
    """


if __name__ == "__main__":
    asyncio.run(main())
    