import os
from pydantic import BaseModel

from autogen import ConversableAgent, LLMConfig, register_function, UserProxyAgent
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
---------------------------------------------------------------------------
In this example, we explore AG2's Agents with the following features:
- Tool usage
- Structured outputs for tools
- Structured outputs for agent responses

This example shows how to create an agent that can use a tool that returns
a response in a structured format. Then, the agent using the tool responds
to the user with a filtered version of the tool output using its own
response model.
---------------------------------------------------------------------------
"""

# 1. Define your structured output model for the tool
class WeatherModel(BaseModel):
    city: str
    current_temperature: float
    temperature_range: str
    humidity: int

# 2. Define a function that returns structured output for the tool
def get_weather(city: str) -> WeatherModel:
    # Simulate fetching weather data
    return WeatherModel(
        city=city,
        current_temperature=22.5,
        temperature_range="20-25C",
        humidity=60
    )

# 3. Create a response model for the UserProxyAgent
# (This filters the other fields from the tool output
#  and only return the city and current temperature to the user)
class WeatherResponseModel(BaseModel):
    city: str
    current_temperature: float

# 4. Create LLM configuration for the weather agent
weather_llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
    response_format=WeatherResponseModel    
)

# 5. Create weather agent with structured output configuration
weather_agent = ConversableAgent(
    name="weather_agent",
    system_message="You provide weather information in a structured format.",
    llm_config=weather_llm_config,
)

# 6. Create LLM configuration for the user agent
user_llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
)

# 7. Create a user proxy agent to interact with the weather agent
with user_llm_config:
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
    )

# 8. Register the weather function with the weather agent
register_function(
    get_weather,
    caller=weather_agent,
    executor=user_proxy,
    description="Get the current weather for a given city",
)

# 9. Initiate a chat with the weather agent
chat_result = user_proxy.initiate_chat(
    recipient=weather_agent,
    message="What's the weather in Tokyo?",
    max_turns=2
)

print(chat_result.chat_history[-1]["content"])
# Expected output: {"city":"Tokyo","current_temperature":22.5}
