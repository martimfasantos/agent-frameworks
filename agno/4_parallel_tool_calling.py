import asyncio
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.sleep import SleepTools
from textwrap import dedent
from settings import settings

# ---------------------------------------------------------
# In this example, we explore Agno's Agent class with the following features:
# - Parallel tool calling (by calling different tool agents in parallel)
# - Pretty-printing responses
#
# More example on: https://github.com/agno-agi/agno/tree/d3cceaf34db84f57d694be3f446a976aaa80a201/cookbook/agent_concepts/async
# ---------------------------------------------------------

model = OpenAIChat(
    id=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    expected_output=dedent("""\
    ## {title}

    {Answer to the user's question}
    """),
    # This will make the agent respond directly to the user, rather than through the team leader.
    respond_directly=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=model,
    tools=[YFinanceTools()],
    instructions="Use tables to display data",
    expected_output=dedent("""\
    ## {title}

    {Answer to the user's question}
    """),
    # This will make the agent respond directly to the user, rather than through the team leader.
    respond_directly=True,
    markdown=True,
)


agent_team = Agent(
    team=[web_agent, finance_agent],
    model=model,
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    debug_mode=True,
    markdown=True,
)

agent_team.print_response(
    '''What's the market outlook and share the latest news for NVDA.
    Get the information in parallel and call the different agents in parallel''',
    stream=True,
    show_full_reasoning=True
)
