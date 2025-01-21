import asyncio
import os
from pyexpat import model
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from settings import settings # not setting the env variables properly
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model=os.getenv("OPENAI_MODEL_NAME"))
    
    assistant = AssistantAgent("assistant", model_client)
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    user_proxy = UserProxyAgent("user_proxy")

    termination = TextMentionTermination("exit") # Type 'exit' to end the conversation.

    # team = RoundRobinGroupChat([web_surfer, assistant, user_proxy], termination_condition=termination)
    # team = SelectorGroupChat(participants=[web_surfer, assistant, user_proxy],
    #                          model_client=model_client, 
    #                          termination_condition=termination)
    team = MagenticOneGroupChat(participants=[web_surfer, assistant, user_proxy],
                                model_client=model_client, 
                                termination_condition=termination)
    
    response = await team.run_stream(task="Find information about AutoGen and write a short summary.")
    print(response)

asyncio.run(main())