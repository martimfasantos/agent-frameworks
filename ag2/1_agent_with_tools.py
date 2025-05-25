import os
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, UserProxyAgent, register_function, LLMConfig
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore AG2's ConversableAgent and UserProxyAgent
with the following features:
- Tool usage
- Function registration
- Agent communication
- Two-way chat

This example shows how to create an agent that can use a tool. This agent is
triggered by an ConversableAgent that serves as an "input" agent.
-----------------------------------------------------------------------------
"""

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

# 1. Define a tool function that will be used by the agent
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

# 2. Agent for determining whether to run the tool
with llm_config:
    date_agent = ConversableAgent(
        name="date_agent",
        system_message="You get the day of the week for a given date.",
    )

# 3. And an agent for executing the tool
executor_agent = ConversableAgent(
    name="executor_agent",
)
# or
executor_agent2 = UserProxyAgent(
    name="executor_agent2",
    human_input_mode="NEVER",
)

# 4. Registers the tool with the agents, the description will be used by the LLM
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# 5. Two-way chat ensures the executor agent follows the suggesting agent
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on the 25th of March 1995, what day was it?",
    max_turns=2,
)

# Expected Workflow:
# - Executor Agent -> Date Agent - get_weekday("1995-03-25") -> Executor Agent

print(chat_result.chat_history[-1]["content"])
# Expected output: "Saturday"