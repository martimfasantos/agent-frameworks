from datetime import date
from tavily import TavilyClient
import json
import asyncio

# Pydantic AI imports
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import Tool
from pydantic_ai import Agent as PydanticAgent, RunContext

# Prompt components
from prompts import role, goal, instructions, knowledge

from utils import get_tools_descriptions

# Load environment variables
from settings import settings

# Initialize Tavily client
tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


class Agent:
    def __init__(self):
        """
        Initialize the Pydantic AI agent.
        """
        self.name = "PydanticAI Agent"

        self.model = OpenAIModel(
            model_name=settings.openai_model_name,
            api_key=settings.openai_api_key.get_secret_value()
        )

        # Create tools
        #   - We dont use dependency injection because we cannot define tool metadata
        self.tools = self._create_tools()
        
        # Create the agent with a comprehensive system prompt
        self.agent = PydanticAgent(
            model=self.model,
            tools=self.tools, # this could be ignored if we used dependency injection
            system_prompt="\n".join([
                role,
                goal,
                instructions,
                "You have access to two primary tools: date and web_search.",
                knowledge
            ]),
            deps_type=str,
            result_type=str
        )

        # Conversation history
        self.messages = []

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.name, tool.description) for tool in self.tools]
        )


    def _create_tools(self):
        """
        Create and register tools for the agent.
        """
        # @self.agent.tool_plain
        async def date_tool() -> str:
            """Get the current date"""
            today = date.today()
            return today.strftime("%B %d, %Y")

        # @self.agent.tool_plain
        async def web_search(query: str) -> str:
            """Search the web for information"""
            # Call Tavily's search and dump the results as a JSON string
            search_response = tavily_client.search(query)
            results = json.dumps(search_response.get('results', []))
            print(f"Web Search Results for '{query}':")
            print(results)
            return results
        
        return [
            Tool(date_tool, name="date_tool", description="Gets the current date"),
            Tool(web_search, name="web_search_tool", description="Searches the web for information")
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
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async function in the loop
            result = loop.run_until_complete(
                self.agent.run(message, deps=message, message_history=self.messages)
            )

            # Close the loop
            loop.close()

            # Maintain conversation history
            self.messages.extend(result.new_messages())

            return result.data

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
            self.messages = []
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """
    agent = Agent()

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
    