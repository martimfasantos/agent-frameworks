import os
from crewai import Crew, Task, Process, LLM
from agents import DataAnalystAgent, TradingAdvisorAgent, TradingStrategyAgent, RiskManagementAgent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from IPython.display import Markdown
from settings import settings # not setting the env variables properly
from dotenv import load_dotenv

load_dotenv()

# Note: this does not follow the structure in https://github.com/crewAIInc/crewAI to set up a project


# llm = ChatGroq(model_name="groq/llama3-8b-8192", # change this to your preferred model
#                temperature=0.5,
#                groq_api_key=os.getenv("GROQ_API_KEY")
#                )
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

# Agents
data_analyst_agent = DataAnalystAgent(llm)
trading_strategy_agent = TradingStrategyAgent(llm)
execution_agent = TradingAdvisorAgent(llm)
risk_management_agent = RiskManagementAgent(llm)

# Tasks
# Task for Data Analyst Agent: Analyze Market Data
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

# Task for Trade Advisor Agent: Plan Trade Execution
execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the "
        "best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

# Task for Risk Advisor Agent: Assess Trading Risks
risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)


# Define the crew with agents and tasks
financial_trading_crew = Crew(
    agents=[data_analyst_agent, 
            trading_strategy_agent, 
            execution_agent, 
            risk_management_agent],
    
    tasks=[data_analysis_task, 
           strategy_development_task, 
           execution_planning_task, 
           risk_assessment_task],
    
    # manager_llm=ChatGroq(model_name="groq/llama3-8b-8192", # change this to your preferred model
    #            temperature=0.5,
    #            groq_api_key=os.getenv("GROQ_API_KEY")),
    # Use if you have an API key for OpenAI
    # manager_llm=ChatOpenAI(model="gpt-4o-mini", 
    #                        temperature=0.7),
    process=Process.sequential, # can be hierarchical with a manager
    verbose=True
)

# Example data for kicking off the process
financial_trading_inputs = {
    'stock_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
