from typing import AsyncGenerator, List, Sequence, Tuple, Callable
from urllib import response
from llama_index.core.agent import ReActAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import ChatMessage, TextMessage
from llama_index.core.tools import FunctionTool, RetrieverTool, ToolMetadata
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


class DatabaseRetriever(BaseChatAgent):
    def __init__(
        self, 
        name: str, 
        description: str, 
        react_agent: ReActAgent,
    ) -> None:
        super().__init__(name, description=description)
        self._react_agent = react_agent
        self._message_history: List[ChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Update the message history.
        # NOTE: it is possible the messages is an empty list, which means the agent was selected previously.
        self._message_history.extend(messages)
        # Execute the other agent
        result = await self._react_agent.achat(messages)
        # Create a new message with the result.
        response_message = TextMessage(content=result.response, source=self.name)
        # Update the message history.
        self._message_history.append(response_message)
        # Return the response.
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass