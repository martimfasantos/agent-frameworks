import sys
import time
import warnings
import logging
import asyncio
import asyncio.log
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from crewai_testing.crew import ChatBot
from pydantic import BaseModel
from crewai_testing.settings import settings


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# set logger and root logger config level
# nice to see the bot "thinking" on the logs
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")


def run():
    """
    Run the crew with a simple input.
    """
    # Ask the user for their query/input
    inputs = {
        "query": "Calculate the geometric mean of the John Doe salary and the Jane Doe salary.",
    }
    ChatBot().crew().kickoff(inputs=inputs)

# ----------------------------------------------

# create a crew instance
crew = ChatBot().crew()

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
        response = await crew.kickoff_for_each_async(inputs=[{"query": q} for q in query])
        times.append(time.time() - start)
        
        if i < settings.num_iterations-1: 
            time.sleep(60) # to make sure we don't hit the token limit
    
    main_logger.info(
        f"Queries took {sum(times)/len(times):.2f} seconds on average to process. \
        (iters={settings.num_iterations})"
    )
    print(crew.usage_metrics)
    
    # --- kickoff_for_each_async (x1) ---
    # 10 inputs - 14.49s
    # 30 inputs - 31.22s (2.16x)
    # 50 inputs - ?? # not able to run given the token limit
    # 100 inputs - ?? # not able to run given the token limit

    # --- kickoff_for_each_async (x10) ---
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
