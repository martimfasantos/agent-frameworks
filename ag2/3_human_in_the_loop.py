import os

from autogen import ConversableAgent, LLMConfig, UserProxyAgent
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
----------------------------------------------------------------------
In this example, we explore AG2's Agents with the following features:
- Human-in-the-loop

This example shows how to create a ConversableAgent / UserProxyAgent 
that can interact with a human user.
----------------------------------------------------------------------
"""

# 1. Define the LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")


# 2. Create the LLM agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

# 3. Create a human agent with manual input mode
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS", # This is important for human-in-the-loop scenarios
    llm_config=llm_config,
)
# or
human = UserProxyAgent(
    name="human", 
    code_execution_config={
        "work_dir": "coding", 
        "use_docker": False
    },
    # human_input_mode="ALWAYS",  # NOTE: this is the default setting
    llm_config=llm_config,
)

# 5. Start the chat
human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
# Expected output:
#   Hello! 2 + 2 equals 4.
#   ------------------------------------------------------------------------------
#   Replying as human. Provide feedback to assistant. Press enter to skip and use 
#   auto-reply, or type 'exit' to end the conversation:
