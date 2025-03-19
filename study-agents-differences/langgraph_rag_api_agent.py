import os
import time

# LangGraph and LangChain imports
from typing import Annotated, TypedDict
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain.tools import Tool, tool
from langchain_core.prompts import ChatPromptTemplate

from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import knowledge, role, goal, instructions

# Tools
from shared_functions import F1API, MetroAPI

# Load environment variables
from settings import settings


# LangGraph specific - Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphRAGandAPIAgent:
    def __init__(
        self, 
        provider: str = "openai", 
        memory: bool = True,
        verbose: bool = False, 
        tokens: bool = False
    ):
        """
        Initialize the LangGraph agent using create_react_agent.
        """
        self.name = "LangGraph RAG & API Agent"

        # Create tools
        self.tools = self._create_tools()

        # Create memory
        if memory:
            self.memory = MemorySaver()
        else:
            self.memory = None
        # Memory will be checkpointed per thread. We will start with thread id 1.
        self.thread_id = 1

        self.tokens = tokens

        # Create the prompt
        self.prompt = self._create_prompt()

        # Initialize the language model
        self.model = (
            AzureChatOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            ) 
            if provider == "azure" and settings.azure_api_key
            else ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                id=settings.openai_model_name,
            )
            if provider == "openai" and settings.openai_api_key
            else ChatHuggingFace(
                model=settings.open_source_model_name
            )
        )

        # Create the agent graph
        self.graph = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.prompt,
            checkpointer=self.memory,
            debug=True if verbose else False
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.name, tool.description) for tool in self.tools]
        )


    @staticmethod
    def load_documents(url: str) -> list[Document]:
        """
        Load all docs from a folder and return them as a list of Document.

        Args:
            urls (list[str]): A list of URLs to load the documents from.
        Returns:
            List[Document]: A list of Document objects.
        """
        return DirectoryLoader(url, glob="*.md", show_progress=True).load()

    @staticmethod
    def create_vectorstore(documents: list[Document]) -> Chroma:
        """
        Create a simple vectorstore using the in memory Chroma
        """
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     model_name=settings.embeddings_model_name,
        #     chunk_size=1024, chunk_overlap=50
        # )
        # doc_splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding= (
                AzureOpenAIEmbeddings(
                    base_url=f"{settings.azure_endpoint}/deployments/{settings.embeddings_model_name}",
                    api_version=settings.embeddings_api_version,
                    api_key=settings.azure_api_key.get_secret_value(),
                ) if settings.azure_api_key else
                FastEmbedEmbeddings()
            ),
            collection_name="local-rag"
        )
        return vectorstore
    
    @staticmethod
    def create_rag_tool():
        """
        RAG tool that loads documents, creates a vectorstore, and returns a retriever tool.
        """
        # Load documents
        docs = LangGraphRAGandAPIAgent.load_documents("knowledge_base/cl_matches/")
        # Create the vectorstore/index        
        vectorstore = LangGraphRAGandAPIAgent.create_vectorstore(docs)

        # Create the retriever tool
        return create_retriever_tool(
            vectorstore.as_retriever(),
            name="RAG_tool",
            description="Search and retrieve information from the knowledge base about the matches of the 2025 UEFA Champions League.",
        )
    
    @tool
    @staticmethod
    def get_driver_info(driver_number: int, session_key: int = 9158) -> str:
        """Useful function to get F1 drivers information."""
        return F1API.get_driver_info(driver_number, session_key)
    
    @tool # this is MUST exists when the tool receives no arguments
    @staticmethod
    def get_state_subway() -> str:
        """Useful function to get state subway information."""
        return MetroAPI.get_state_subway()
    
    @tool
    @staticmethod
    def get_times_next_two_subways_in_station(station: str) -> str:
        """Useful to get the time (in seconds) of the next two subways in a station."""
        return MetroAPI.get_times_next_two_subways_in_station(station)
    
    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            # RAG tool
            self.create_rag_tool(),
            # API tools - MetroAPI and F1API - Created using @tool
            self.get_driver_info,
            self.get_state_subway,
            self.get_times_next_two_subways_in_station
        ]

    def _create_prompt(self):
        """
        Create a comprehensive prompt for the agent.

        Returns:
            ChatPromptTemplate
        """
        return ChatPromptTemplate.from_messages([
            ("system", "\n".join([knowledge, role, goal, instructions])),
            ("placeholder", "{messages}"),
        ])

    def _inc_thread_id(self):
        """
        Simply increments the thread id and returns the new id.

        """
        new_thread_id = self.thread_id + 1
        self.thread_id = new_thread_id
        return new_thread_id

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Prepare input
            inputs = {"messages": [("user", message)]}
            config = {"configurable": {"thread_id": str(self.thread_id)}}


            # Stream the graph updates and collect the final response
            full_response = ""
            start = time.perf_counter()
            for event in self.graph.stream(inputs, config=config, stream_mode="values"):
                if event and "messages" in event:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content"):
                        full_response = last_message.content
            end = time.perf_counter()
            exec_time = end - start
            
            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": 0,
                    "completion_llm_token_count": 0,
                    "total_llm_token_count": 0,
                }
                for message in event["messages"]: # last event contains all messages
                    if message.response_metadata:
                        token_usage = message.response_metadata["token_usage"]
                        tokens["prompt_llm_token_count"] += token_usage["prompt_tokens"]
                        tokens["completion_llm_token_count"] += token_usage["completion_tokens"]
                        tokens["total_llm_token_count"] += token_usage["total_tokens"]
            else:
                tokens = {}

            return full_response, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request."

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            self._inc_thread_id() # Incrementing the thread ID basically resets the memory
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """
    
    args = parse_args()

    agent = LangGraphRAGandAPIAgent(
        provider=args.provider,
        memory=False if args.no_memory else True,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"]
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
    