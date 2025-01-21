from crewai import Agent
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
# import tavil


search_tool = SerperDevTool() # paid
scrape_tool = ScrapeWebsiteTool()

# -------------------------------------
#            CrewAI Example
# -------------------------------------

class DataAnalystAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Data Analyst",
            goal="Monitor and analyze market data in real-time "
                 "to identify trends and predict market movements.",
            backstory="Specializing in financial markets, this agent "
                      "uses statistical modeling and machine learning "
                      "to provide crucial insights. With a knack for data, "
                      "the Data Analyst Agent is the cornerstone for "
                      "informing trading decisions.",
            verbose=True,
            allow_delegation=False,
            # tools = [scrape_tool, search_tool],
            llm=llm,
        )

class TradingStrategyAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Trading Strategy Developer",
            goal="Develop and test various trading strategies based "
                 "on insights from the Data Analyst Agent.",
            backstory="Equipped with a deep understanding of financial "
                      "markets and quantitative analysis, this agent "
                      "devises and refines trading strategies. It evaluates "
                      "the performance of different approaches to determine "
                      "the most profitable and risk-averse options.",
            verbose=True,
            allow_delegation=False,
            # tools = [scrape_tool, search_tool],
            llm=llm,
        )

class TradingAdvisorAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Trade Advisor",
            goal="Suggest optimal trade execution strategies "
                "based on approved trading strategies.",
            backstory="This agent specializes in analyzing the timing, price, "
                    "and logistical details of potential trades. By evaluating "
                    "these factors, it provides well-founded suggestions for "
                    "when and how trades should be executed to maximize "
                    "efficiency and adherence to strategy.",
            verbose=True,
            allow_delegation=False,
            # tools = [scrape_tool, search_tool],
            llm=llm,
        )

class RiskManagementAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Risk Advisor",
            goal="Evaluate and provide insights on the risks "
                "associated with potential trading activities.",
            backstory="Armed with a deep understanding of risk assessment models "
                    "and market dynamics, this agent scrutinizes the potential "
                    "risks of proposed trades. It offers a detailed analysis of "
                    "risk exposure and suggests safeguards to ensure that "
                    "trading activities align with the firm’s risk tolerance.",
            verbose=True,
            allow_delegation=False,
            # tools = [scrape_tool, search_tool],
            llm=llm,
        )


# -------------------------------------
#    CrewAI Example with Human Tools
# -------------------------------------
import chainlit as cl
from chainlit import run_sync
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import BaseTool
from crewai.tools import tool

# human_tools = load_tools(["human"]) # not working because of pydantic error

@tool("Ask Human follow up questions")
def ask_human(question: str) -> str:
    """Ask human follow up questions"""
    human_response  = run_sync( cl.AskUserMessage(content=f"{question}").send())
    if human_response:
        return human_response["output"]

class EmailWriter(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Concise Email Writer",
            goal="Write a short and engaging email",
            backstory="Experienced in writing concise marketing emails.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

class DTCCMOAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="DTC CMO",
            goal="Lead the team in creating effective cold emails",
            backstory="A CMO who frequently receives marketing emails and knows what stands out.",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            # Passing human tools to the agent
            # tools=[ask_human] # not working because of pydantic error
        )

class CopywriterAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Professional Copywriter",
            goal="Critique and refine the email content",
            backstory="A professional copywriter with extensive experience in persuasive writing.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
