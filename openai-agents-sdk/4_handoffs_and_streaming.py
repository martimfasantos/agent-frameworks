import os
import asyncio
import uuid

from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
--------------------------------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Routing / Handoffs
- Streaming
- Tracing

This example shows the handoffs/routing pattern. The orchestrator agent receives 
the first message, and then hands off to the appropriate agent based on the 
language of the request. Responses are streamed to the user.
--------------------------------------------------------------------------------
"""

# 1. Define the language expert agents
french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
    handoff_description="A french speaking agent",
)

portuguese_agent = Agent(
    name="portuguese_agent",
    instructions="You only speak Portuguese",
    handoff_description="A portuguese speaking agent",
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
    handoff_description="An english speaking agent",
)

# 2. Define the orchestrator agent that routes to the appropriate language expert
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, portuguese_agent, english_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    msg = input("Hi! We speak French, Portuguese and English. How can I help? ")
    agent = orchestrator_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        with trace("4_handoffs_and_streaming", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")

        if user_msg.lower() == "exit":
            print("Exiting...")
            break

        inputs.append({"content": user_msg, "role": "user"})
        # Re use the agent from the last result
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())