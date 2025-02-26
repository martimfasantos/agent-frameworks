from datetime import date
from scipy.__config__ import show
from tavily import TavilyClient
import json

# Llama-Index imports
from agno.models.openai import OpenAIChat
from agno.agent import Agent as AgnoAgent
from agno.memory import AgentMemory
from agno.tools import tool
from agno.tools.tavily import TavilyTools

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge

# Load environment variables
from settings import settings

# # Initialize Tavily client - Not needed, we can leverage TavilyTools directly
# tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


class Agent:
    def __init__(self, memory: bool = True):
        """
        Initialize the Agno agent.
        """
        self.name = "Agno Agent"

        self.model = OpenAIChat(
            api_key=settings.openai_api_key.get_secret_value(),
            id=settings.openai_model_name,
        )
        
        # Create tools
        self.tools = self._create_tools()

        # Create the Agent
        self.agent = AgnoAgent(
            name="Agno Agent",
            # role="Search the web for information",
            model=self.model,
            tools=self.tools,
            # instructions="Always include sources",
            system_message="\n".join([
                role,
                goal,
                instructions,
                "You have access to two primary tools: date and web_search.",
                knowledge
            ]),
            memory=AgentMemory(),
            add_history_to_messages=True if memory else False,
            read_chat_history=True if memory else False,
            respond_directly=True,
            markdown=True,
            # show_tool_calls=True # to show the tools calls in the response
        )

        # Extras: 
        self.tools_descriptions = get_tools_descriptions(
            [(tool.name, getattr(tool, 'description', str(tool))) for tool in self.tools]
        )



    @staticmethod
    @tool(name="date_tool", description="Gets the current date")
    def date_tool():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    # This tool is part of the TavilyTools toolkit, we can leverage it directly
    @staticmethod
    @tool(name="web_search_tool", description="Searches the web for information")
    def web_search_tool(query: str):
        """
        This function searches the web for the given query and returns the results.
        """
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
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
            self.date_tool,
            # self.web_search
            TavilyTools(api_key=settings.tavily_api_key.get_secret_value()), # <-- we can levarage the integrated ToolKit directly
            # NOTE: this won't show the print in the console, but it will return the tool call in the response
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
            response = self.agent.run(message)

            return response.content

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

    if args.no_memory:
        agent = Agent(memory=False)
    else:
        agent = Agent()

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
    