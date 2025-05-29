import os
import asyncio
from pydantic import BaseModel

from agents import Agent, Runner, RunResult, function_tool
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Tool usage
- Output models
- Internal messages
- Metrics

This example shows the parallelization pattern. We run the agent 
three times in parallel, and pick the best result.
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
    
    # 4. Run the agent
    result: RunResult = await Runner.run(agent, input="What's the weather in Tokyo?")
    print("-" * 50)
    print(f"Final response:\n\t{result.final_output}")
    
    """ 
    Expected output: 
        The weather in Tokyo is sunny with wind, and 
        the temperature ranges from 14°C to 20°C.
    """
    
    # 5. Print the internal messages and metrics
    print("-" * 50)
    print("Internal messages:")
    for item in result.new_items:
        print(f"\t{item}")
    print("-" * 50)
    print("Metrics:")
    for raw_response in result.raw_responses:
        print(f"\t{raw_response.usage}")
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())