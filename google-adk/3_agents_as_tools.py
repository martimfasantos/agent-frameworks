import os
import asyncio

from google.adk.tools import agent_tool
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor

from utils import call_agent_async
from settings import settings


"""
-------------------------------------------------------
In this example, we explore Google's ADK agents with the following features:
- Tool usage
- Agent as a Tool
- Code Execution
- Orchestrator Agent

This example shows how to create two agents with built-in tools:
1. A search agent that uses the Google Search tool.
2. A coding agent that uses the Built-In Code Executor tool.

And an orchestrator agent that uses both agents as tools to answer a question.
-------------------------------------------------------
"""

# 1. Create the agents that will be used as tools
search_agent = Agent(
    name='search_agent',
    model='gemini-2.0-flash',
    instruction=(
        "You're a specialist in Google Search"
    ),
    tools=[google_search],
)

coding_agent = Agent(
    name='coding_agent',
    model='gemini-2.0-flash',
    instruction=(
        "You're a specialist in Code Execution"
    ),
    code_executor=[BuiltInCodeExecutor],
)

# 2. Create the orchestrator agent that uses the above agents as tools
orchestrator = Agent(
    name="orchestrator",
    model="gemini-2.0-flash",
    description="Orchestrator Agent",
    tools=[
        agent_tool.AgentTool(agent=search_agent), 
        agent_tool.AgentTool(agent=coding_agent)
    ],
)

# 3. Run the orchestrator agent
input = "What is the current price of Bitcoin in USD? Write a Python code to fetch it."
print("Input: ", input)
asyncio.run(
    call_agent_async(orchestrator, input, tool_calls=True, tool_call_results=True)
)