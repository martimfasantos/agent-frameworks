import os
import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore OpenAI's Agent class with the following features:
- Agents as tools
- Tracing

This example shows the agents-as-tools pattern. The frontline agent receives 
a user message and then picks which agents to call, as tools. In this case, 
it picks from a set of translation agents.
-----------------------------------------------------------------------------
"""

# 1. Define the agents that will be used as tools
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An english to italian translator",
)

# 2. Define the orchestrator agent that uses the translation agents as tools
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[  # Pass the agents as tools here
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
)

# 3. Define the synthesizer agent that will concatenate the results
synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions=(
        f"You inspect translations, correct them if needed, and "
        f"produce a final concatenated response.",
    )    
)


async def main():
    msg = input("Hi! What would you like translated, and to which languages? ")

    # 4. Run the agents with tracing
    with trace("5_agents_as_tools"):
        # 4.1 Run the orchestrator agent to get translations
        orchestrator_result = await Runner.run(orchestrator_agent, msg)

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")

        # 4.2 Run the synthesizer agent to combine the translations
        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())