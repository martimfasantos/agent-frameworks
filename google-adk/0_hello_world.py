import os
import asyncio

from google.adk.agents import LlmAgent

from utils import call_agent_async
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
-------------------------------------------------------
"""

# 1. Define the agent - the main agent should be named `root_agent`
hello_world_agent = LlmAgent( # or use Agent() if you prefer
    name="hello_world_agent",
    instruction="You are a helpful assistant", # this is the system prompt
    model='gemini-2.0-flash',
    description="An AI assistant that can help you.",
)

# 2. Run the agent
asyncio.run(call_agent_async(hello_world_agent, "Hello"))

# NOTE: Runner.run_async is the primary method for executing agent invocations. 
# All core runnable components (Agents, specific flows) 
# use asynchronous methods internally.