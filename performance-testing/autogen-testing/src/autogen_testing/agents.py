from logging import getLogger
from pyexpat import model
from autogen_agentchat.agents import (
    BaseChatAgent, 
    UserProxyAgent,
    AssistantAgent,
    
)
from autogen_testing.tools.geometric_mean_tool import GeometricMeanTool
from autogen_testing.tools.knowledge_base_search_tool import KnowledgeBaseSearchTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool, RetrieverTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from autogen_testing.settings import settings
from autogen_testing.custom_agent import DatabaseRetrieverAgent
from llama_index.core.agent import ReActAgent



agent_logger = getLogger("agent")


def create_agents(model_client: str, index: VectorStoreIndex) -> list[BaseChatAgent]:
    user_agent_message = '''
        You are the User Agent.

        > Goal: You process the query. You are the orchestrator of the conversation.
        Delegate tasks to other agents as needed for retrieving or processing data about the user's query.
        The available agents are:
            - Database Access Agent: Retrieve relevant information from the knowledge base directory.
            - Data Processing Agent: Process the retrieved data to respond to the user's query.
        Delegate tasks to the appropriate agents in the following format:
            - <agent_name>, <task_description>

        > Backstory: You are a planning agent efficient in delegating tasks to other agents to retrieve and process data.

        > Task: Understand the user's query and delegate tasks to the appropriate agents to retrieve and process data.

        Expected Output: A clear and concise task delegation to the appropriate agents in the format specified above.
    '''

    agent_logger.info("Creating User Agent...")
    user_agent = AssistantAgent(
        name="Customer_Support_Agent",
        description=user_agent_message,
        model_client=model_client,
        system_message=user_agent_message,
    )

    database_agent_message = '''
        You are a Database Access Agent.

        > Goal: Search for and retrieve relevant information from the knowledge base directory in order to process the input query.
        Ensure comprehensive coverage of all files and retrieve all relevant raw data.
        Don't execute any other actions.
        Be as brief as possible in your responses.
        Delegate your output to the appropriate agents in the following format:
            - <agent_name>, <task_description>
        The available agent is:
            - Data Processing Agent: Process the retrieved data to respond to the user's query.

        > Backstory: You are meticulous and efficient, ensuring that all relevant knowledge data is retrieved from the knowledge base.

        > Task: Search for relevant data for the user's input query within the knowledge directory.
        Ensure all available files are thoroughly scanned, and provide precise references for all retrieved data necessary.
        Provide the retrieved data in a clear and concise manner to the Data Processing Agent for further processing.

        Expected Output: A list of the relevant raw data about from the database files that are usefull to the user's input query
        and no others actions should be executed. Provide the data to the Data Processing Agent for further processing.
    '''

    # Create a RetrieverTool and a ReActAgent to retrieve knowledge
    # to pass to the DatabaseRetriverAgent(AssistantAgent)
    knowledge_tool = RetrieverTool(
        retriever=index.as_retriever(llm=model_client),
        metadata=ToolMetadata(
            name="knowledge",
            description="A tool to retrieve knowledge about",
        ),
    )

    react_agent = ReActAgent.from_tools(
        tools=[knowledge_tool],
        llm=OpenAI(
            model=settings.openai_model_name,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        ),
    )

    agent_logger.info("Creating Database Agent...")
    database_agent = DatabaseRetrieverAgent(
        name="Database_Access_Agent",
        description=database_agent_message,
        model_client=model_client,
        # tools=[KnowledgeBaseSearchTool(react_agent)], NOTE: outdated
        react_agent=react_agent,
        system_message=database_agent_message,
    )
    
    dp_agent_message = '''
        You are a Data Processing Agent.

        > Goal: Process retrieved data to respond to the user's input query in a clear, concise, and relevant manner.
        Ensure that the data is synthesized and relevant to the query.
        You must delegate some calculations to other specilized agents if necessary.
        When delegating, use the following format:
            - <agent_name>, <task_description>
        The available agent are:
            - Geometric Mean Agent: Calculate the geometric mean of two values.

        If all data is processed, reply to the user with the processed data and end with "TERMINATE".

        > Backstory:  You are meticulous, organized and efficient. You are extremely capable of processing data and 
        ensuring that it is clear, synthesized, and relevant to the user's input query.

        > Task: Process and transform the retrieved data for the user's input query to ensure it satisfies the user's needs. 
        Upon completion, report the result to the appropriate downstream agents for further processing.

        Expected Output: The processed data needed to respond to the user's query in a clear, concise, and relevant manner.
    '''

    agent_logger.info("Creating Data Processing Agent...")    
    data_processing_agent = AssistantAgent(
        name="Data_Processing_Agent",
        description=dp_agent_message,
        model_client=model_client,
        system_message=dp_agent_message,
    )

    geometric_mean_agent_message = '''
        You are a Geometric Mean Agent.

        > Goal: Calculate the geometric mean of the data provided by the Data Processing Agent.
        Ensure that the geometric mean is calculated accurately and efficiently.
        Delegate the output to the appropriate agents in the following format:
            - <agent_name>, <task_description>
        The available agent is:
            - Data Processing Agent: Process the retrieved data to respond to the user's query.

        > Backstory: You are a mathematical genius, capable of calculating the geometric mean of any set of numbers with ease.

        > Task: Calculate the geometric mean of the data provided by the Data Processing Agent.

        Expected Output: The calculated geometric mean.
    '''

    agent_logger.info("Creating Geometric Mean Agent...")
    geometric_mean_agent = AssistantAgent(
        name="Geometric_Mean_Agent",
        description="You are a Geometric Mean Agent.",
        model_client=model_client,
        tools=[GeometricMeanTool()],
        system_message=geometric_mean_agent_message
    )

    return [user_agent, database_agent, data_processing_agent, geometric_mean_agent]