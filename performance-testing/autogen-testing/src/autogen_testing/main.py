import time
import logging
import asyncio
import asyncio.log
from autogen_core import CancellationToken
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from autogen_agentchat.teams import (
    SelectorGroupChat,
    RoundRobinGroupChat,
    Swarm,
    MagenticOneGroupChat,
)
from autogen_testing.agents import create_agents
from autogen_testing.settings import settings
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

# set logger and root logger config level
# nice to see the bot "thinking" on the logs
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")


model_client = OpenAIChatCompletionClient(
                    model=settings.openai_model_name,
                    api_key=settings.openai_api_key.get_secret_value(),
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )

# Create the agents
agents = create_agents(model_client)
user_agent = agents[0] # hardcoded: user agent is the first one in the list

# aux: clear agents' "memories" 
for agent in agents:
    agent.on_reset(CancellationToken())

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
    queries = request.query
    times = []
    for i in range(settings.num_iterations):
        start = time.time()
        results = await asyncio.gather(*(groupchat.run(task=query) for query in queries))
        times.append(time.time() - start)
        
        if i < settings.num_iterations-1: 
            main_logger.info("Sleeping for 60 seconds to avoid token limit...")
            time.sleep(60) # to make sure we don't hit the token limit
    
    main_logger.info("---------------------------------------")
    main_logger.info(
        f"Queries took {sum(times)/len(times):.2f} seconds on average to process. \
        (iters={settings.num_iterations}) \n \
        Last Task Results: \n \
        \t {results[-1]}"
    )

    # --- .run(x1) ---
    # 10 inputs - 12.71s
    # 30 inputs - 31.22s (2.16x)
    # 50 inputs - ?? # not able to run given the token limit
    # 100 inputs - ?? # not able to run given the token limit

    # --- .run(x10) ---
    # 10 inputs - avg: -- 16.49s
    #   - total_tokens=47178, prompt_tokens=44223, cached_prompt_tokens=0, completion_tokens=2955, successful_requests=72
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


if __name__ == "__main__":
    # launch uvicorn and serve
    asyncio.run(serve())