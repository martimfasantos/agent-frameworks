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


class Orchestrator:
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = cast(
            Router, self.llm.with_structured_output(Router).invoke(self.prompt.format(messages=messages))
        )
        return {"message": response}


class GeneralQuestionAgent:
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = self.llm.invoke(self.prompt.format(messages=messages))
        return {"messages": response}


class AskMoreInfoAgent:
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", MORE_INFO_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = self.llm.invoke(self.prompt.format(messages=messages))
        return {"messages": response}


class RAGAgent:
    def __init__(self, llm: ChatOpenAI = AgentsConfiguration.llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Execute RAG tool and when you have the results, generate a response."),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        result = self.llm.invoke(self.prompt.format(
            messages=messages
        ))
        print(result)
        # If tool was called, execute the tool and process the result
        if result.tool_calls:
            return {"messages": result}
        
        else:
            return {"messages": "parse the result"}
            # If tools are invoked, LLM will handle it automatically.
            tool_result = result.tool_calls[0]
            print("-----")
            print(tool_result)
            print("-----")
            parse_tool_result = self.llm.invoke(self.prompt.format(
                query=messages, tool_result=tool_result
            ))
            # print(parse_tool_result)
            return {"messages": parse_tool_result}

        return {"messages": result}
    