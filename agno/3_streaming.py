import asyncio
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from settings import settings

# ---------------------------------------------------------
# In this example, we explore Agno's Agent class with the following features:
# - Streaming responses
# ---------------------------------------------------------

async def main():

    agent = Agent(
        model=OpenAIChat(
            id=settings.openai_model_name,
            api_key=settings.openai_api_key.get_secret_value(),
        ),
        tools=[DuckDuckGoTools()],
        description="You are an enthusiastic news reporter with a flair for storytelling!",
        show_tool_calls=True,
        markdown=True,
    )

    # result = await agent.arun("Tell me about a breaking news story from New York.", stream=False)
    result = await agent.arun("Tell me about a breaking news story from New York.", stream=True)
    async for r in result:
        print(r.content) # Print the tokens - results yielded are RunResponse() dataclass objects


if __name__ == "__main__":
    asyncio.run(main())