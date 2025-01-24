import os
import time
import logging
import asyncio
import asyncio.log
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from autogen_agentchat.teams import (
    SelectorGroupChat,
    GroupChatManager,
)
from .agents import create_agents
from dotenv import load_dotenv
from .settings import settings

load_dotenv()

# set logger and root logger config level
# nice to see the bot "thinking" on the logs
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")

config_list = [
    {
        "model": settings.openai_model_name,
        "api_key": settings.openai_api_key,
    }
]

llm_config = {
    "config_list": config_list, 
    "temperature": settings.temperature,
    "timeout": 60, 
    "seed": 42
}

# Create the agents
agents = create_agents(llm_config, settings.openai_model_name)
user_agent = agents[0] # hardcoded: user agent is the first one in the list

#  -- we want to leverage the agents' memory --
# aux: clear agents' "memories" 
# for agent in agents:
#     agent.clear_memory()

groupchat = SelectorGroupChat(
    agents=agents,
    messages=[], 
    max_round=5, 
    speaker_selection_method="auto",
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
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
    query = request.query
    times = []
    for i in range(settings.num_iterations):
        start = time.time()
        response = await user_agent.a_initiate_chats(
            [
                {
                    "recipient": "user_agent",
                    "message": "This is the user's input query: " + query,
                } 
                for query in query
            ],
        )
        times.append(time.time() - start)
        
        if i < settings.num_iterations-1: 
            time.sleep(60) # to make sure we don't hit the token limit
    
    main_logger.info(
        f"Queries took {sum(times)/len(times):.2f} seconds on average to process. \
        (iters={settings.num_iterations})"
    )
    
    # --- a_initiate_chats (x1) ---
    # 10 inputs - 14.49s
    # 30 inputs - 31.22s (2.16x)
    # 50 inputs - ?? # not able to run given the token limit
    # 100 inputs - ?? # not able to run given the token limit

    # --- a_initiate_chats (x10) ---
    # 10 inputs - avg: -- 16.49s
    #   - total_tokens=47178, prompt_tokens=44223, cached_prompt_tokens=0, completion_tokens=2955, successful_requests=72
    # 30 inputs - avg: --
    # 50 inputs - ?? # not able to run given the token limit
    # 100 inputs - ?? # not able to run given the token limit

    return JSONResponse(content={"response": [r.raw for r in response]})


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


if __name__ == "__main__":
    # launch uvicorn and serve
    asyncio.run(serve())