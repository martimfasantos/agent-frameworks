from typing import Literal
from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph_project.state import AgentState
from langgraph_project.agents.agent import Assistant
from langgraph_project.agents.configuration import AgentsConfiguration
from langgraph_project.vector_store.index import create_index
from langgraph_project.tools import (
    retrieve_information_vectorbase,
    analyze_and_route_query,
    respond_to_general_query,
    ask_user_for_more_info,
)
from langgraph_project.utils import create_tool_node_with_fallback
from langgraph.prebuilt import tools_condition
from langgraph_project.settings import settings


# Check https://github.com/langchain-ai/rag-research-agent-template
# and https://github.com/martimfasantos/agent-frameworks/tree/main/langgraph/langgraph-examples


# Initialize the VectorStore and set the index
index = create_index()
retrieve_information_vectorbase = partial(retrieve_information_vectorbase, index=index)


# Define the graph
builder = StateGraph(AgentState)

# Orchestrator
builder.add_node("orchestrator", Assistant(
    AgentsConfiguration.router_system_prompt |
    AgentsConfiguration.llm.bind_tools(analyze_and_route_query)
))
# RAG agent
builder.add_node("rag_agent", Assistant(
    AgentsConfiguration.execute_rag_system_prompt | 
    AgentsConfiguration.llm.bind_tools(retrieve_information_vectorbase)
))
# Ask for more info agent
builder.add_node("ask_user_more_info_agent", Assistant(
    AgentsConfiguration.more_info_system_prompt |
    AgentsConfiguration.llm.bind_tools(ask_user_for_more_info)
))
# Responds to a general question (no RAG)
builder.add_node("respond_to_general_question_agent", Assistant(
    AgentsConfiguration.general_system_prompt |
    AgentsConfiguration.llm.bind_tools(respond_to_general_query)
))
# Tools nodes
builder.add_node("analyze_and_route_query", create_tool_node_with_fallback(analyze_and_route_query))
builder.add_node("retrieve_information_vectorbase", create_tool_node_with_fallback(retrieve_information_vectorbase))
builder.add_node("ask_for_more_info", create_tool_node_with_fallback(ask_user_for_more_info))
builder.add_node("respond_to_general_query", create_tool_node_with_fallback(respond_to_general_query))


# Edges
builder.add_edge(START, "orchestrator")

def route_agent(
    state: AgentState
) -> Literal["rag_agent", "ask_user_more_info_agent", "respond_to_general_question_agent"]:
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    _type = state.router["type"]
    if _type == "langchain":
        return "rag_agent"
    elif _type == "more-info":
        return "ask_user_more_info_agent"
    elif _type == "general":
        return "respond_to_general_question_agent"
    else:
        raise ValueError(f"Unknown router type {_type}")

builder.add_conditional_edges("orchestrator", route_agent)
builder.add_edge("retrieve_information_vectorbase", END)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
