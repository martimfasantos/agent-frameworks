from logging import getLogger
from autogen_agentchat.agents import (
    BaseChatAgent, 
    UserProxyAgent,
    AssistantAgent,
    
)
from autogen_testing.tools.geometric_mean_tool import GeometricMeanTool

agent_logger = getLogger("agent")


def create_agents(model_client: str) -> list[BaseChatAgent]:
    user_agent_message = '''
        You are a Customer Support Agent.
        > Goal: Help by answering their queries clearly and concisely.
        Delegate tasks to other agents as needed for retrieving or processing data about the user query.
        > Backstory: You're friendly and knowledgeable, ensuring that user feels supported. 
        You always strive to provide accurate and helpful responses about the query.
        > Task: Engage in a conversation with the user to fully understand their query.
        Employ active listening and clarifying questions to address any ambiguities or incompleteness.
        Determine the most appropriate actions and data sources to effectively retrieve the necessary information
        Expected Output: A clear and concise response that addresses the user's query.
        If the query is unclear, provide a helpful and informative request for clarification.
        Reply with TERMINATE when the task has been completed.
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
        > Backstory: You are meticulous and efficient, ensuring that all relevant knowledge data is retrieved from the knowledge base.
        > Task: Search for relevant data for the user's input query within the knowledge directory.
        Ensure all available files are thoroughly scanned, and provide precise references for all retrieved data necessary.
        Provide the retrieved data in a clear and concise manner to the Data Processing Agent for further processing.
        Expected Output: A list of the relevant raw data about from the database files that are usefull to the user's input query
        and no others actions should be executed.
    '''

    agent_logger.info("Creating Database Agent...")
    database_agent = AssistantAgent(
        name="Database_Access_Agent",
        description=database_agent_message,
        model_client=model_client,
        tools=[], # TODO add Tool to search and retrieve data from database -> What tool?
        system_message=database_agent_message,
    )

    dp_agent_message = '''
        You are a Data Processing Agent.
        > Goal: Process retrieved data to respond to the user's input query in a clear, concise, and relevant manner.
        Ensure that the data is synthesized and relevant to the query.
        Ensure comprehensive coverage of all files.
        Be as brief as possible in your responses.
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
        tools=[GeometricMeanTool()],
        system_message=dp_agent_message,
    )

    # We could add a UserProxyAgent to simulate a user input or to have a human in the loop,
    # but since we are performing automated testing, we will not include it in this case.

    return [user_agent, database_agent, data_processing_agent]