import os
from agents import Agent, Runner
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
-------------------------------------------------------
"""

# 1. Define the agent
agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant", # this is the system prompt
    model="gpt-4o-mini",  # specify the model to use (default: "gpt-4o")
)

# 2. Run the agent with a user message
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)