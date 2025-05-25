import os

from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from settings import settings

"""
---------------------------------------------------------------
In this example, we explore a simple Assistant agent.

The interaction with this agent is triggered by a UserProxyAgent 
that serves as an "input" agent.
This agent interaction pattern is characteristic of AG2.
Check out the documentation for more details:
https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/conversable-agent/
---------------------------------------------------------------
"""

# 1. Set the OpenAI API key from the settings
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

# 3. Create an AssistantAgent instance with the LLM configuration
with llm_config:
    assistant = AssistantAgent("assistant")

# 4. Create a UserProxyAgent instance
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 5. Initiate a chat with the assistant using the user proxy
user_proxy.initiate_chat(
    assistant, 
    message="Plot a chart of NVDA and TESLA stock price change YTD."
)