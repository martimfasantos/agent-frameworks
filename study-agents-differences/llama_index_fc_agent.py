import os
import sys
import logging
import time
import tiktoken
from datetime import date
from litellm import api_base, azure_embedding_models
from openai import azure_endpoint
from tavily import TavilyClient
import json


# Llama-Index imports
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent import FunctionCallingAgent, FunctionCallingAgentWorker
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PromptTemplate

# Prompt components
from prompts import role, goal, instructions, knowledge
from prompts import llama_index_react_prompt # extra import

from utils import get_tools_descriptions, parse_args, execute_agent

# Load environment variables
from settings import settings

# Initialize Tavily client
tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)

class Agent:
    def __init__(
        self, 
        provider: str = "openai", 
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False
    ):
        """
        Initialize the Llama-Index agent.
        """
        self.name = "Llama-Index Function Calling Agent"
        
        # Initialize the language model
        self.model = (
            AzureOpenAI(
                engine=settings.azure_deployment_name,
                api_base=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
                callback_manager=CallbackManager([token_counter]) if tokens else None
            )
            if provider == "azure" and settings.azure_api_key
            else OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                id=settings.openai_model_name,
            )
            if provider == "openai" and settings.openai_api_key
            else HuggingFaceInferenceAPI(
                model=settings.open_source_model_name
            )
        )

        # Create tools
        self.tools = self._create_tools()


        # Create the agent
        self.agent = FunctionCallingAgentWorker.from_tools(
            llm=self.model,
            tools=self.tools,
            max_function_calls=2,
            system_prompt="\n".join([
                role,
                goal,
                instructions,
                "You have access to two primary tools: date_tool and web_search_tool.",
                knowledge,
                # llama_index_react_prompt
            ]),
            verbose=True if verbose else False
        ).as_agent()

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.metadata.name, tool.metadata.description) for tool in self.tools]
        )



    @staticmethod
    def date_tool():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    def web_search_tool(query: str):
        """
        This function searches the web for the given query and returns the results.
        """
        # Call Tavily's search and dump the results as a JSON string
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get('results', []))
        # print(f"Web Search Results for '{query}':")
        # print(results)
        return results

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            FunctionTool.from_defaults(
                fn=self.date_tool,
                name="date_tool",
                description="Useful for getting the current date"
            ),
            FunctionTool.from_defaults(
                fn=self.web_search_tool,
                name="web_search_tool",
                description="Useful for searching the web for information"
            )
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
            response = self.agent.chat(message)
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": token_counter.total_embedding_token_count,
                    "prompt_llm_token_count": token_counter.prompt_llm_token_count,
                    "completion_llm_token_count": token_counter.completion_llm_token_count,
                    "total_llm_token_count": token_counter.total_llm_token_count
                }
                token_counter.reset_counts()
            else:
                tokens = {}

            return str(response), exec_time, tokens

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
            self.agent.reset()
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """

    args = parse_args()

    agent = Agent(
        provider=args.provider,
        memory=False if args.no_memory else True,
        verbose=args.verbose
    )

    execute_agent(agent, args)

if __name__ == "__main__":
    main()
    