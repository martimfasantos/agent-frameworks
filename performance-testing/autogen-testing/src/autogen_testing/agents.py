from logging import getLogger
from autogen_agentchat.agents import (
    BaseChatAgent, 
    UserProxyAgent,
    AssistantAgent,
)
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

agent_logger = getLogger("agent")


def create_agents(llm_config: dict, model: str) -> list[BaseChatAgent]:
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
    '''

    agent_logger.info("Creating User Agent...")
    user_agent = UserProxyAgent(
        name="Customer Support Agent",
        description=user_agent_message,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config=False,  # we don't want to execute code in this case.
        system_message=user_agent_message,
    )

    database_agent_message = '''
        You are a Database Access Agent.
        > Goal: Search for and retrieve relevant information from the knowledge base directory for a input query.
        Ensure that you only retrieve the information needed for the user query.
        Be as brief as possible in your responses.
        Ensure comprehensive coverage of all files and return results with appropriate references.
        > Backstory: You are meticulous, organized and efficient, ensuring that all relevant data is retrieved for the user input query.
        You work closely with other agents to fulfill user queries.
        > Task: Search for relevant information for the user's input query within the knowledge directory.
        Ensure all available files are thoroughly scanned, and provide precise references for all retrieved data necessary.
        Expected Output: A list of the relevant information about from the database files that are usefull to the user's input query.
    '''

    agent_logger.info("Creating Database Agent...")
    database_agent = RetrieveUserProxyAgent(
        name="Database Access Agent",
        description=database_agent_message,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        retrieve_config={
            "task": "default",
            "docs_path": "./knowledge",
            "vector_db": "chroma",
            "collection_name": "groupchat",
            # "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "model": model,
            "get_or_create": True,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
        system_message=database_agent_message,
    )

    dp_agent_message = '''
        You are a Data Processing Agent.
        > Goal: Process retrieved data to respond to the user's input query in a clear, concise, and relevant manner.
        Ensure that the data is synthesized and relevant to the query.
        Be as brief as possible in your responses.
        Ensure comprehensive coverage of all files and return results with appropriate references.
        > Backstory:  You are meticulous, organized and efficient. You are extremely capable of processing data and 
        ensuring that it is clear, synthesized, and relevant to the user's input query.
        > Task: Process and transform the retrieved data for the user's input query to ensure it satisfies the user's needs. 
        Upon completion, report the result to the appropriate downstream agents for further processing.
        Expected Output: The processed data needed to respond to the user's query in a clear, concise, and relevant manner.
    '''

    agent_logger.info("Creating Data Processing Agent...")    
    data_processing_agent = AssistantAgent(
        name="Data Processing Agent",
        description=dp_agent_message,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        llm_config=llm_config,
        system_message=dp_agent_message,
    )

    return [user_agent, database_agent, data_processing_agent]