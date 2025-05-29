import os
import asyncio
from pydantic import BaseModel

from agents import Agent, Runner, trace, function_tool
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Tracing
- Function tools
- Output models
- Agent tool calling

This example shows how to use tracing to monitor the execution of an agent workflow.
This allows for better debugging and understanding of the agent's behavior.
Tracing captures LLM calls, handoffs, tool calls, execution, and more.
-----------------------------------------------------------------------------
"""

# 1. Define the output model for the weather data
class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str

# 2. Define a function tool to get weather information
@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")

# 3. Define the agent with the function tool registered
agent = Agent(
    name="Example Agent",
    instructions="You are a helpful agent.",
    tools=[get_weather],    # Register the function in the agent
)

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    # 4. Run the agent with tracing
    with trace("8_tracing"):
        # Because the two calls to `Runner.run` are wrapped in a with `trace()`, 
        # the individual runs will be part of the overall trace rather than creating two traces.
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")

    """
    NOTE: Please check the `traces/` directory for the traces of this and the other examples.
    You can find your traces on: https://platform.openai.com/traces
    """

if __name__ == "__main__":
    asyncio.run(main())
