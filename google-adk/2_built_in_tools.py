
import os
import asyncio

from google.adk.agents import Agent
from google.adk.tools import google_search

from utils import call_agent_async
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google's ADK agents with the following features:
- Tool usage
- Built-in Tools

This example shows how to create and run an agent with a 
built-in tool to perform Google Search.

> NOTE: 
> LIMITATIONS:
    Currently, for each root agent or single agent, 
    only one built-in tool is supported. 
    No other tools of any type can be used in the same agent.
-------------------------------------------------------
"""

# 1. Define the agent with built-in tool
search_agent = Agent(
    name='search_agent',
    model='gemini-2.0-flash',
    instruction=(
        "You're a specialist in Google Search"
    ),
    tools=[
        google_search
        # NOTE: Only one built-in tool is supported per agent.
    ],
)

# 2. Run the agent
input = "Current price of Bitcoin in USD"
print("Input: ", input)
asyncio.run(
    call_agent_async(search_agent, input, tool_calls=True, tool_call_results=True)
)