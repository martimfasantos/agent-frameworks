import asyncio
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
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


    # Define the Summary and Vector Index over the data
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    # Define Query Engines and Set Metadata
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    # Define the Router Query Engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )

    # Run the Query Engine
    # Summary Index
    response = query_engine.query("What is the summary of the document?")
    print(str(response))
    """
    Selecting query engine 0: The question asks for a summary of the document, which aligns with the purpose of choice 1, as it is useful for summarization questions..
    
    The document introduces MetaGPT, a meta-programming framework designed for multi-agent collaboration using large language models (LLMs)...
    """
    print(len(response.source_nodes)) # all nodes were touched since it used the summary index
    print("-"*50)


    # Vector Index
    response = query_engine.query(
        "How do agents share information with other agents?"
    )
    print(str(response))
    """
    Selecting query engine 1: The question pertains to specific mechanisms of information sharing among agents, which would likely be detailed in the MetaGPT paper..

    Agents share information with other agents through a shared message pool, where they can publish structured messages and ...
    """


if __name__ == "__main__":
    asyncio.run(main())