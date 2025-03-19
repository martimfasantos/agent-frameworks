import os
import time
from datetime import date

# Agno imports
from agno.models.openai import OpenAIChat
from agno.models.azure import AzureOpenAI
from agno.agent import Agent as AgnoAgent
from agno.memory import AgentMemory
from agno.tools import tool
from agno.models.huggingface import HuggingFace
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.azure import AzureOpenAI
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.document.base import Document
from agno.document.chunking.fixed import FixedSizeChunking
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.memory import AgentMemory

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import knowledge, role, goal, instructions

# Tools
from shared_functions import F1API, MetroAPI

# Load environment variables
from settings import settings


class AgnoRAGandAPIAgent:
    def __init__(
        self, 
        provider: str = "openai", 
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False
    ):
        """
        Initialize the Agno agent.
        """
        self.name = "Agno RAG & API Agent"

        self.model = ( #    NOTE: available in v1.1.8 after the PR: https://github.com/agno-agi/agno/pull/2273
            AzureOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            ) 
            if provider == "azure" and settings.azure_api_key 
            else 
            OpenAIChat(
                api_key=settings.openai_api_key.get_secret_value(),
                id=settings.openai_model_name,
            ) 
            if provider == "openai" and settings.openai_api_key
            else HuggingFace(
                model_name=settings.open_source_model_name,
            )
        )

        # Create tools
        self.tools = self._create_tools()

        # Load documents
        docs = self.load_documents_from_folder("knowledge_base/cl_matches")
        # Create the knowledge base
        knowledge_base = self.create_knowledge_base(docs)

        knowledge_base.load(recreate=True)

        # Create the Agent
        self.agent = AgnoAgent(
            name="Agno Agent",
            model=self.model,
            tools=self.tools,
            knowledge=knowledge_base, # <-- RAG is passed here as knowledge base
            # Add a tool to search the knowledge base which enables agentic RAG.
            # This is enabled by default when `knowledge` is provided to the Agent.
            # instructions="Always include sources",
            system_message="\n".join([
                knowledge,
                role,
                goal,
                instructions,
                "You have access to the knowledge base about the matches of the 2025 UEFA Champions League "
                "and the provided tools to get information about F1 drivers, the state of the subway, and the times "
                "of the next two subways in a station.",
            ]),
            memory=AgentMemory(), # <-- even if memory is None, it will still be created when the agent runs
            add_history_to_messages=True if memory else False,
            read_chat_history=True if memory else False,
            respond_directly=True,
            markdown=True,
            # to show the tools calls in the response
            show_tool_calls=True if verbose else False
        )

        self.tokens = tokens

        # Extras: 
        self.tools_descriptions = get_tools_descriptions(
            [("RAG_tool", "Knowledge base for the 2025 UEFA Champions League matches")] \
            + [(self.get_date.name, self.get_date.description)] \
            + F1API.list_functions() \
            + MetroAPI.list_functions()
        )

    @staticmethod
    def load_documents_from_folder(docs_path: str) -> list[Document]:
        """
        Load all docs from a folder and return them as a list of Document.

        Args:
            docs_path: Path to the folder containing the documents
        Returns:
            List of documents
        """
        documents = []
        for file_name in os.listdir(docs_path):
            if file_name.endswith(".md"):
                file_path = os.path.join(docs_path, file_name)
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    documents.append(
                        Document(content=content, meta_data={"file_name": file_name})
                    )
        return documents
    
    @staticmethod
    def create_knowledge_base(documents: list[Document]) -> DocumentKnowledgeBase:
        """
        Create a knowledge base from a list of documents.

        Args:
            documents: List of documents
        Returns:
            DocumentKnowledgeBase
        """
        vector_db = ChromaDb(
            embedder = (
                AzureOpenAIEmbedder(
                    base_url=f"{settings.azure_endpoint}/deployments/{settings.embeddings_model_name}",
                    api_version=settings.embeddings_api_version,
                    api_key=settings.azure_api_key.get_secret_value(),
                )
                if settings.azure_api_key
                else f"local:{settings.local_embeddings_model_name}"
            ),
            collection="local-rag"
        )

        chunking_strategy = FixedSizeChunking(
            chunk_size=1024, overlap=50
        )

        return DocumentKnowledgeBase(
            documents=documents,
            vector_db=vector_db,
            chunking_strategy=chunking_strategy,
            num_documents=2
        )
    

    @staticmethod
    @tool(name="date_tool", description="Gets the current date")
    def get_date():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")


    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            # RAG tool
            # NOTE: Not passed here, passed as knowledge base to the agent
            # Date tool
            self.get_date,
            # API tools
            F1API.get_driver_info,
            MetroAPI.get_state_subway,
            MetroAPI.get_times_next_two_subways_in_station
        ]

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Send message to the agent
            start = time.perf_counter()
            response = self.agent.run(message)
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": sum(response.metrics["prompt_tokens"]),
                    "completion_llm_token_count": sum(response.metrics["completion_tokens"]),
                    "total_llm_token_count": sum(response.metrics["total_tokens"]),
                }
            else:
                tokens = {}

            return response.content, exec_time, tokens

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
            # Reset the agent's chat history
            self.agent.memory.clear()
            return True
        except Exception as e:
            print(f"Error in clearing memory: {e}")
            return False


def main():

    """
    Example usage demonstrating the agent interface.
    """

    args = parse_args()

    agent = AgnoRAGandAPIAgent(
        provider=args.provider,
        memory=False if args.no_memory else True,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"]
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
    