import os
import asyncio
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nest_asyncio

from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from dotenv import load_dotenv

load_dotenv()


# Define an asynchronous function that simulates some asynchronous task (e.g., I/O operation)
async def my_asynchronous_function():
    print("Simulating an asynchronous task...")
    await asyncio.sleep(2)
    return "human input" 
    # return "exit" # this will stop the execution
    # return input("Enter your input: ") # <-- Provide human feedback here (this is not async)


# Define a custom class `CustomisedUserProxyAgent` that extends `UserProxyAgent`
class CustomisedUserProxyAgent(UserProxyAgent):
    # Asynchronous function to get human input
    async def a_get_human_input(self, prompt: str) -> str:
        # Call the asynchronous function to get user input asynchronously
        user_input = await my_asynchronous_function()

        return user_input

    # Asynchronous function to receive a message (can be customized as needed)
    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)


class CustomisedAssistantAgent(AssistantAgent):
    # Asynchronous function to get human input
    async def a_get_human_input(self, prompt: str) -> str:
        # Call the asynchronous function to get user input asynchronously
        user_input = await my_asynchronous_function()

        return user_input

    # Asynchronous function to receive a message
    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)



llm_config = {
    "config_list": [{"model": os.getenv("OPENAI_MODEL_NAME")}], 
    "temperature": 0.5,
    "timeout": 60, 
    "cache_seed": 42
}


async def main():

    boss = CustomisedUserProxyAgent(
        name="Boss",
        human_input_mode="ALWAYS", # <-- Set to "ALWAYS" to always request human input for the Boss agent
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = CustomisedAssistantAgent(
        name="Assistant",
        system_message='''
            You will provide some agenda, and I will create questions for an interview meeting. 
            Every time when you generate question then you have to ask user for feedback and if 
            user provides the feedback then you have to incorporate that feedback and generate 
            new set of questions and if user don't want to update then terminate the process and exit
            ''',
        llm_config=llm_config,
    )

    await boss.a_initiate_chat(
        assistant,
        message="Resume Review, Technical Skills Assessment, Project Discussion, Job Role Expectations, Closing Remarks.",
        n_results=3,
    )



if __name__ == "__main__":
    # nest_asyncio.apply()
    asyncio.run(main())