from typing import AsyncGenerator, List, Sequence, Tuple, Callable
from urllib import response
from llama_index.core.agent import ReActAgent
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.messages import HandoffMessage, ToolCallSummaryMessage
from rich.console import Console
from rich.markdown import Markdown
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResultMessage,
    UserMessage,
    SystemMessage
)


class DatabaseRetrieverAgent(AssistantAgent):
    def __init__(self, *args, react_agent: ReActAgent, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._react_agent = react_agent
        # define the knowledge base search as a tool of this agent
        self._tools.append(
            FunctionTool(
                self._search_knowledge_base, name="search_knowledge_base", 
                description="Call this to search the knowledge base for needed information."
            )
        )

    async def _search_knowledge_base(self, query: str) -> str:
        result = await self._react_agent.achat(query)
        return result.response
    