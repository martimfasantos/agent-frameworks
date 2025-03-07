from datetime import date
from json import tool
from tavily import TavilyClient
import json

# Llama-Index imports
from openai import OpenAI, AzureOpenAI

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge
from prompts import openai_completion_after_tool_call_prompt

# Load environment variables
from settings import settings

# # Initialize Tavily client - Not needed, we can leverage TavilyTools directly
# tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


class Agent:
    def __init__(
        self, 
        provider: str = "openai", 
        memory: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the OpenAI agent.
        """
        self.name = "OpenAI Agent"

        self.model = (
            AzureOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            ) 
            if provider == "azure" and settings.azure_api_key 
            else 
            OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                id=settings.openai_model_name,
            ) 
            if provider == "openai" and settings.openai_api_key
            else None
        )

        # Create tools
        self.tools = self._create_tools()

        # Create prompt
        self.prompt = self._create_prompt()

        # Create the Agent
        self.agent = self.model.chat.completions

        self.verbose = verbose



    @staticmethod
    def date_tool():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    # This tool is part of the TavilyTools toolkit, we can leverage it directly
    @staticmethod
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
    
    def call_function(self, name, args):
        if name == "date_tool":
            return self.date_tool()
        if name == "web_search_tool":
            return self.web_search_tool(**args)

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "date_tool",
                    "description": "Useful for getting the current date.",
                },
                "strict": True,
                "parameters": {  # Even if there are no parameters, an empty object is expected
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search_tool",
                    "description": "Useful for searching the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Provide a query to search the web for information."
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
    
    def _create_prompt(self):
        """
        Create a prompt for the agent.

        Returns:
            str: Prompt for the agent
        """
        return {
            "role": "system",
            "content": f"{role}\n{goal}\n{instructions}\n{knowledge}",
        }

    def chat(self, message):
        """
        Send a message and get a completion.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:

            messages = [
                self.prompt,
                {"role": "user", "content": message}
            ]

            # Send prompt + user_message to the agent
            completion = self.agent.create(
                model=(
                    settings.azure_deployment_name if self.model == AzureOpenAI
                    else settings.openai_model_name if self.model == OpenAI
                    else None
                ),
                messages=messages,
                tools=self.tools
            )

            response_message = completion.choices[0].message
            tool_calls = response_message.tool_calls

            # Agent returns tool calls, we need to process them and manually call the functions
            if tool_calls:
                messages.append(response_message)

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if self.verbose:
                        print(f"Tool call name: {name}")
                        print(f"Tool call args: {args}")

                    # Call the chosen tool
                    chosen_tools = eval(f"self.{name}")
                    tool_result = chosen_tools(**args)
                    
                    # Append the tool result to the messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result  
                    })

            # Append a prompt to the messages to get the final completion
            messages.append(
                {
                    "role": "system",
                    "content": openai_completion_after_tool_call_prompt
                }
            )

            # Send the messages with the tool results back to the agent to get the final completion
            completion2 = self.agent.create(
                model=(
                    settings.azure_deployment_name if self.model == AzureOpenAI
                    else settings.openai_model_name if self.model == OpenAI
                    else None
                ),
                messages=messages,
            )

            return completion2.choices[0].message.content

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

    agent = Agent(
        provider=args.provider,
        memory=False if args.no_memory else True,
        verbose=args.verbose
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
    