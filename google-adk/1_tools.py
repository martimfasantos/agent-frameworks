import os
import asyncio
import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from utils import call_agent_async
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google's ADK agents with the following features:
- Tool usage
- Multiple custom Tools

This example shows how to create an agent with two custom function tools:
1. A tool to get weather information for a specified city.
2. A tool to get the current time in a specified city.
-------------------------------------------------------
"""

# 1. Define a custom function tools
# 1.1 Define the function to get weather information
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "lisbon":
        return {
            "status": "success",
            "report": (
                "The weather in Lisbon is sunny with a temperature of 25 degrees"
                " Celsius."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }
        
weather_tool = FunctionTool(func=get_weather)

# 1.2 Define the function to get the current time in a specified city
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

current_time_tool = FunctionTool(func=get_current_time)


# 2. Register the customs FunctionTools in an agent (or the directly the functions)
weather_time_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about weather in a city and "
        "search the web."
    ),
    instruction=(
        "You are a helpful assistant that can provide weather information "
        "for a city and search the web for information."
    ),
    tools=[
        weather_tool,       # Created tool for weather information
        current_time_tool,  # Created tool for current time
    ],
    # tools=[get_weather, get_current_time],
    # NOTE: Alternatively, you can use the tools directly without FunctionTool
)

# 3. Run the agent
input = "Tell me the weather in Lisbon."
print("Input: ", input)
asyncio.run(
    call_agent_async(weather_time_agent, input, tool_calls=True)
)
# > Output: The weather in Lisbon is sunny with a temperature of 25 degrees Celsius.

input = "What is the current time in New York?"
print("Input: ", input)
asyncio.run(
    call_agent_async(weather_time_agent, input, tool_calls=True)
)
# > Output: The current time in New York is 2025-06-28 12:00:00 EDT-0400
