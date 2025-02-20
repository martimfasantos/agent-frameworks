import json
from textwrap import dedent
from typing import Iterator

import httpx
from agno.agent import Agent
from agno.exceptions import StopAgentRun
from agno.tools import FunctionCall, tool
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Prompt
from agno.models.openai import OpenAIChat
from settings import settings

# ---------------------------------------------------------
# In this example, we explore Agno's Agent class with the following features:
# - Human-in-the-loop (pre-hooks on tool calling)
# - Streaming outputs from tools
#
# This example is from: https://github.com/agno-agi/agno/blob/d3cceaf34db84f57d694be3f446a976aaa80a201/cookbook/getting_started/19_human_in_the_loop.py
# ---------------------------------------------------------


""" Human-in-the-Loop: Adding User Confirmation to Tool Calls

This example shows how to implement human-in-the-loop functionality in your Agno tools.
It shows how to:
- Add pre-hooks to tools for user confirmation
- Handle user input during tool execution
- Gracefully cancel operations based on user choice

Some practical applications:
- Confirming sensitive operations before execution
- Reviewing API calls before they're made
- Validating data transformations
- Approving automated actions in critical systems
"""

model = OpenAIChat(
    id=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)

# This is the console instance used by the print_response method
# We can use this to stop and restart the live display and ask for user confirmation
console = Console()


def pre_hook(fc: FunctionCall):
    print("Pre-hook triggered!") 
    # Get the live display instance from the console
    live = console._live

    # Stop the live display temporarily so we can ask for user confirmation
    live.stop()

    # Ask for confirmation
    console.print(f"\nAbout to run [bold blue]{fc.function.name}[/]")
    message = (
        Prompt.ask("Do you want to continue?", choices=["y", "n"], default="y")
        .strip()
        .lower()
    )

    # Restart the live display
    live.start()

    # If the user does not want to continue, raise a StopExecution exception
    if message != "y":
        raise StopAgentRun(
            "Tool call cancelled by user",
            agent_message="Stopping execution as permission was not granted.",
        )


@tool(pre_hook=pre_hook)
def get_top_hackernews_stories(num_stories: int) -> Iterator[str]:
    """Fetch top stories from Hacker News after user confirmation.

    Args:
        num_stories (int): Number of stories to retrieve

    Returns:
        str: JSON string containing story details
    """
    # Fetch top story IDs
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # Yield story details
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        yield json.dumps(story)


# Initialize the agent with a tech-savvy personality and clear instructions
agent = Agent(
    description="A Tech News Assistant that fetches and summarizes Hacker News stories",
    model=model,
    instructions=dedent("""\
        You are an enthusiastic Tech Reporter

        Your responsibilities:
        - Use the tool to present Hacker News stories in an engaging and informative way
        - Provide clear summaries of the information you gather

        Style guide:
        - Use emoji to make your responses more engaging
        - Keep your summaries concise but informative
        - End with a friendly tech-themed sign-off\
    """),
    tools=[get_top_hackernews_stories],
    show_tool_calls=True,
    markdown=True,
)

# Example questions to try:
# - "What are the top 3 HN stories right now?"
# - "Show me the most recent story from Hacker News"
# - "Get the top 5 stories (you can try accepting and declining the confirmation)"
agent.print_response(
    "Get the top 5 stories from Hacker News using the provided tool", stream=True, console=console
)

# View all messages
pprint(agent.run_response.messages)