import os
import chromadb
from typing_extensions import Annotated

from autogen import (
    UserProxyAgent,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
)
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("OPENAI_MODEL_NAME")

llm_config = {
    "config_list": [{"model": model}], 
    "temperature": 0.5,
    "timeout": 60, 
    "seed": 42
}

URL = "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md"
PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."


# aux
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


# Boss != Manager - In this case, Boss is a proxy agent for the user, that can execute code and provide feedback to the other agents
boss = UserProxyAgent(
    name="Boss",
    system_message="The boss who ask questions and give tasks.",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who ask questions and give tasks.",
)

# Boss Assistant - The Retrieval-Augmented User Proxy retrieves document chunks based on the embedding
# similarity, and sends them along with the question to the Assistant
boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": URL,
        "chunk_token_size": 1000,
        "model": model,
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
    description="Assistant who has extra content retrieval power for solving difficult problems.",
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message='''
        You are a senior python engineer, you provide python code to answer questions.
        ''',
    llm_config=llm_config,
    description="Senior Python Engineer who can write code to solve problems and answer questions.",
)

pm = AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message='''
        You are a product manager.
        ''',
    llm_config=llm_config,
    description="Product Manager who can design and plan the project.",
)

reviewer = AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message='''
        You are a code reviewer. You review the code and provide feedback about the code quality.
        The code reviewer is responsible for ensuring the code quality.
        ''',
    llm_config=llm_config,
    description="Code Reviewer who must review the code for further improvements.",
)


# aux: clear agents' "memories"
def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


# ------------------------------------
#            NO RAG CHAT
# ------------------------------------
def norag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss, pm, coder, reviewer], # boss is on the groupchat
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )

# ------------------------------------
#             RAG CHAT
# ------------------------------------
def rag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss_aid, pm, coder, reviewer], # boss is not on the groupchat
        messages=[], 
        max_round=12, 
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        message=PROBLEM,
        n_results=3,
    )


# ------------------------------------
#           CALL RAG CHAT
# ------------------------------------

def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        _context = {"problem": message, "n_results": n_results}
        ret_msg = boss_aid.message_generator(boss_aid, None, _context)
        return ret_msg or message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    for agent in [pm, coder, reviewer]:
        d_retrieve_content = agent.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [boss, pm]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = GroupChat(
        agents=[boss, pm, coder, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM +" Must retrieve content using function calling.",
    )


if __name__ == "__main__":
    # norag_chat()
    rag_chat()
    # call_rag_chat()