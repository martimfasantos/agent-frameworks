import sys
import time
import logging
import asyncio
import asyncio.log
from statistics import stdev
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from autogen_agentchat.teams import (
    SelectorGroupChat,
    RoundRobinGroupChat,
    Swarm,
    MagenticOneGroupChat,
)
from autogen_agentchat.ui import Console
from autogen_project.loader import load_documents_from_folder
from autogen_project.agents import create_agents
from autogen_project.index import create_index
from autogen_project.settings import settings
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

logging.basicConfig(level=logging.INFO) 
logging.getLogger().setLevel(logging.INFO)

# remove all unnecessary logs
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen.events").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat.events").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")
# main_logger.setLevel(logging.INFO)

model_client = OpenAIChatCompletionClient(
                    model=settings.openai_model_name,
                    api_key=settings.openai_api_key.get_secret_value(),
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )

# Load the documents
documents = load_documents_from_folder(settings.knowledge_base_path)

# Create the index
index = create_index(documents)

# Create the agents
agents = create_agents(model_client, index)
user_agent = agents[0] # hardcoded: user agent is the first one in the list

groupchat = SelectorGroupChat(     # this GroupChat only allows for ChatAgents. Some operations
    participants=agents,           # might be more efficient if we use other types of agents.
    model_client=model_client,     # e.g. ToolAgent, to execute tool calls.
    termination_condition=TextMentionTermination("TERMINATE"), # Important! - otherwise the chat will never end
    allow_repeated_speaker=True,
)

# ----------------------------------------------

# create a an api using fast api
app = FastAPI()


# use pydantic to map the body parameters of the post request bellow
class QueryRequest(BaseModel):
    query: list[str]


@app.post("/query")
async def query_agent(request: QueryRequest):
    """
    Handle incoming chat requests asynchronously.
    """
    async def run_with_new_groupchat(task):
        groupchat = SelectorGroupChat(
            participants=create_agents(model_client, index),
            model_client=model_client,
            termination_condition=TextMentionTermination("TERMINATE"),
            allow_repeated_speaker=True,
        )
        return await groupchat.run(task=task)
    
    queries = request.query
    times = []
    for i in range(settings.num_iterations):
        start = time.time()
        results = await asyncio.gather(
            *(run_with_new_groupchat(task=query) for query in queries)
            # NOTE: we have to create a new groupchat with new agents for each query
        )
        times.append(time.time() - start)
        
        if i < settings.num_iterations-1:
            main_logger.info(
                f"Completed iteration: {i+1}/{settings.num_iterations}, \
                took {times[-1]:.2f} seconds to process.\n \
                Sleeping for 60 seconds to avoid token limit..."
            )
            time.sleep(60) # to make sure we don't hit the token limit
    
    print(
        f'''
        Queries took {sum(times)/len(times):.2f} ± {stdev(times) if len(times) > 1 else 0} 
        seconds on average to process. (iters={settings.num_iterations}) \n
        Last Task Results: \n\t {results[-1]}
        '''
    )

    # --- Performance Results ---
    #       (iterations=10)
    # 1 input - 16.21 ± 5.20 seconds
    # 10 inputs - 20.69 ± 6.97 seconds
    # 30 inputs - avg: --
    # 50 inputs - ?? # not able to run given the token limit
    # 100 inputs - ?? # not able to run given the token limit

    return JSONResponse(content={"response": [r.messages[-1].content for r in results]})


async def serve():
    """
    Launch the uvicorn single worker
    """
    import uvicorn

    config = uvicorn.Config(
        app, host=settings.api_host, port=settings.api_port, reload=True
    )
    server = uvicorn.Server(config)
    main_logger.info("")
    main_logger.info("#" * 32)
    main_logger.info(f"Go to http://{settings.api_host}:{settings.api_port}/docs")
    main_logger.info("#" * 32)
    await server.serve()


async def streamlit_chat():
    import streamlit as st

    with st.chat_message("user"):
        st.write("Hello!")

    prompt = st.chat_input("Say something")
    if prompt:
        result = await groupchat.run(task=prompt)
        st.write("Agent: ", (message for message in result.messages))



if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        asyncio.run(streamlit_chat())
    else:
        # launch uvicorn and serve
        asyncio.run(serve())