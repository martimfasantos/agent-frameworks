import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_example.settings import settings

async def main():
    agent = AssistantAgent(
        name = "assistant", 
        model_client = OpenAIChatCompletionClient(
            model=settings.openai_model_name, 
            api_key=settings.openai_api_key.get_secret_value()
        )
    )
    result = await agent.run(task="Say 'Hello World!'")
    print(result.messages[1].content)

asyncio.run(main())