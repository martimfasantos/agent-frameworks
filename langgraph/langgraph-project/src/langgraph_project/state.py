"""State management for the retrieval graph.

This module defines the state structures used in the retrieval graph. It includes
definitions for agent state, input state, and router classification schema.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class AgentState(TypedDict):
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    

class Router(TypedDict):
    """Classify user query."""

    logic: str
    type: Literal["more-info", "user", "general"]
