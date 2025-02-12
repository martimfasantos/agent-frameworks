import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph_project.state import AgentState
from langgraph_project.agents.agents import (
    Orchestrator,
    GeneralQuestionAgent,
    AskMoreInfoAgent,
    RAGAgent,
    GenerateResponseAgent
)
from langgraph_project.agents.configuration import AgentsConfiguration
from langgraph_project.utils import create_tool_node_with_fallback, save_graph_image, _print_event
from langgraph_project.tools import retrieve_information_vectorbase
from langgraph_project.settings import settings


logging.basicConfig(level=logging.INFO) 
logging.getLogger().setLevel(logging.INFO)

# remove all unnecessary logs
logging.getLogger("httpx").setLevel(logging.WARNING)


main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")


# Define the graph
builder = StateGraph(AgentState)

# Orchestrator
builder.add_node("orchestrator", Orchestrator())
# Ask for more info agent
builder.add_node("ask_user_more_info_agent", AskMoreInfoAgent())
# Responds to a general question (no RAG)
builder.add_node("respond_to_general_question_agent", GeneralQuestionAgent())
# RAG agent - with the tool for retrieving information
builder.add_node("rag_agent", RAGAgent(
    AgentsConfiguration.llm.bind_tools([retrieve_information_vectorbase])
))
# Tools nodes
builder.add_node(
    "retrieve_information_vectorbase", 
    create_tool_node_with_fallback([retrieve_information_vectorbase])
)
# Generate responses after executing RAG
builder.add_node("generate_response_agent", GenerateResponseAgent())

# Edges
builder.add_edge(START, "orchestrator")

def route_agent(
    state: AgentState
) -> Literal["rag_agent", "ask_user_more_info_agent", "respond_to_general_question_agent"]:
    _type = state["message"]["type"]
    if _type == "user":
        return "rag_agent"
    elif _type == "more-info":
        return "ask_user_more_info_agent"
    elif _type == "general":
        return "respond_to_general_question_agent"
    else:
        raise ValueError(f"Unknown router type {_type}")

builder.add_conditional_edges(
    "orchestrator", route_agent, ["rag_agent", "ask_user_more_info_agent", "respond_to_general_question_agent"]
)
builder.add_edge("rag_agent", "retrieve_information_vectorbase")
builder.add_edge("retrieve_information_vectorbase", "generate_response_agent")

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"

save_graph_image(graph, "./img", "graph.png")

# result = graph.invoke({"messages": [("user", "How much does John Doe pay for his service?")]})

# print(result)
_printed = set()
events = graph.stream(
    {"messages": ("user", "What is John Doe's account number?")}, stream_mode="values" # RAG
    # {"messages": ("user", "Can you tell me John's number?")}, stream_mode="values" # More info
    # {"messages": ("user", "Which stocks showed the most growth in the last 5 years?")}, stream_mode="values" # General

)

for event in events:
    _print_event(event, _printed)