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

# model = HfApiModel(model_id="meta-llama/Llama-3.2-1B-Instruct") # not so good
# model = HfApiModel(model_id="mistralai/Codestral-22B-v0.1") # doesn't have the chat/completion endpoint
model = HfApiModel()
# model = LiteLLMModel(model_id="gpt-4o-mini-mini") # use if you have a api key

# ------------ example with multi-agent ------------

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=10,
)

managed_web_agent = ManagedAgent( # Adds additional prompting for the managed agent, like adding more context to the query
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# Setup
# CodeAgent <- ManagedAgent <- ToolCallingAgent <- DuckDuckGoSearchTool

# answer = manager_agent.run("What year was the movie 'Rebel Without a Cause' released and who was the star of it?")
answer = manager_agent.run("What movie was released in 1955 that stars James Dean and what is it's Rotten Tomatoes score?")

print(answer)
