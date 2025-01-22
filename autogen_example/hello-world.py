import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from settings import settings # not setting the env variables properly
from dotenv import load_dotenv

load_dotenv()

async def main():
    agent = AssistantAgent(
            name = "assistant", 
            model_client=OpenAIChatCompletionClient(model=os.getenv("OPENAI_MODEL_NAME"))
        )
    result = await agent.run(task="Say 'Hello World!'")
    print(result.messages[1].content)

asyncio.run(main())