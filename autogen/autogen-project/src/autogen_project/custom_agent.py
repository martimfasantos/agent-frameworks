from llama_index.core.agent import ReActAgent
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_core.tools import FunctionTool

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
    