from operator import index
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, RetrieverTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from autogen_testing.settings import settings


# ----------------------------------------------
#                   NOT USED
# ----------------------------------------------

class KnowledgeBaseSearchArgs(BaseModel):
    query: str
    index: str


class KnowledgeBaseSearchResult(BaseModel):
    response: str


class KnowledgeBaseSearchTool(BaseTool[KnowledgeBaseSearchArgs, KnowledgeBaseSearchResult]):
    def __init__(self, react_agent: ReActAgent) -> None:
        super().__init__(
            args_type=KnowledgeBaseSearchArgs,
            return_type=KnowledgeBaseSearchResult,
            name="Knowledge_Base_Search_Tool",
            description="Searches a knowledge/data base for relevant information.",
        )
        self._react_agent = react_agent

    async def run(self, args: KnowledgeBaseSearchArgs, cancellation_token: CancellationToken) -> KnowledgeBaseSearchResult:
        result = await self._react_agent.achat(args.query)
        return KnowledgeBaseSearchResult(response=result.response)