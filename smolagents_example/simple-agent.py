from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    tool
)
from settings import settings # not setting the env variables properly
from dotenv import load_dotenv

load_dotenv()

model = HfApiModel()
# model = LiteLLMModel(model_id="gpt-4o-mini-mini") # use if you have a api key

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

response = agent.run("What is the capital of France?")

print(response)