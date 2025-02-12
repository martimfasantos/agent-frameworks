"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from langchain_openai import ChatOpenAI
import langgraph_project.agents.prompts as prompts
from langgraph_project.settings import settings

class AgentsConfiguration:
    """The configuration for the agents."""

    # models (only one model is used in this example)

    llm: ChatOpenAI = ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    # prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for asking for more information from the user."
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

    execute_rag_system_prompt: str = field(
        default=prompts.EXECUTE_RAG_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses after executing RAG."},
    )

    