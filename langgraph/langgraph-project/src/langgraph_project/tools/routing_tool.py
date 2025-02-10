from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig

from langgraph_project.agents.configuration import AgentConfiguration
from langgraph_project.state import AgentState, Router
from langgraph_project.utils import load_chat_model


# TODO

# Only responsible for assigning the correct key for the router
async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}

