from logging import getLogger
from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph_project.state import AgentState, Router
from langgraph_project.agents.configuration import AgentsConfiguration
from langgraph_project.agents.prompts import (
    EXECUTE_RAG_SYSTEM_PROMPT, 
    GENERAL_SYSTEM_PROMPT, 
    MORE_INFO_SYSTEM_PROMPT, 
    ROUTER_SYSTEM_PROMPT
)
from langchain_core.messages import ToolMessage

agent_logger = getLogger("agents")

class BaseAgent:
    def __init__(self, system_prompt: str, llm: ChatOpenAI = AgentsConfiguration.llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: AgentState, config: RunnableConfig):
        agent_logger.info(f"{"-" * 12} {self.__class__.__name__} responding ... {"-" * 12}")
        messages = state["messages"]
        response = self.llm.invoke(self.prompt.format(messages=messages))
        return {"messages": response}


class Orchestrator(BaseAgent):
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        super().__init__(ROUTER_SYSTEM_PROMPT, llm)

    def __call__(self, state: AgentState, config: RunnableConfig):
        agent_logger.info(f"{"-" * 12} {self.__class__.__name__} responding ... {"-" * 12}")
        messages = state["messages"]
        response = cast(
            Router, self.llm.with_structured_output(Router).invoke(self.prompt.format(messages=messages))
        )
        return {"message": response}
    

class GeneralQuestionAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        super().__init__(GENERAL_SYSTEM_PROMPT, llm)


class AskMoreInfoAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        super().__init__(MORE_INFO_SYSTEM_PROMPT, llm)


class RAGAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        super().__init__("You are a helpful customer support assistant for Mobile Operator. "
            "You are an expert programmer and problem-solver, tasked with answering any user's query. Use the provided \
tools to help you find the information you need."
            " Use the provided tools to search for user's information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up.", llm)
        
    def __call__(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]

        if isinstance(messages[-1], ToolMessage):
            return {"messages": messages[-1]}

        result = self.llm.invoke(self.prompt.format(messages=messages))
        return {"messages": result}


class GenerateResponseAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        super().__init__(EXECUTE_RAG_SYSTEM_PROMPT, llm)

    def __call__(self, state: AgentState, config: RunnableConfig):
        agent_logger.info(f"{"-" * 12} {self.__class__.__name__} responding ... {"-" * 12}")
        messages = state["messages"]
        # previous message was a tool message with the results
        tool_results = messages[-1].content
        # second to last message was the tool call with the query
        query = messages[-2].tool_calls[0]["args"]["query"]
        response = self.llm.invoke(self.prompt.format(
            messages=messages, 
            query=query,
            tool_results=tool_results))
        return {"messages": response}
        
    