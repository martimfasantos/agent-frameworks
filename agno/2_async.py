import asyncio
import time
from agno.agent import Agent
from agno.cli.console import console
from agno.models.openai import OpenAIChat
from settings import settings

# ---------------------------------------------------------
# In this example, we explore Agno's Agent class with the following features:
# - Asynchronous runs 
# - Synchronous runs
# ---------------------------------------------------------

task = "9.11 and 9.9 -- which is bigger?"

model = OpenAIChat(
    id=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)

regular_agent = Agent(
    model=model, 
)
reasoning_agent = Agent(
    model=model,
    structured_outputs=True,
)

# Synchronous execution function
def run_sync():
    start_time = time.time()
    
    regular_response = regular_agent.run(task)
    reasoning_response = reasoning_agent.run(task)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return regular_response, reasoning_response, duration

# Asynchronous (parallel) execution function
async def run_parallel():
    start_time = time.time()
    
    regular_task = asyncio.create_task(regular_agent.arun(task))
    reasoning_task = asyncio.create_task(reasoning_agent.arun(task))
    
    regular_response, reasoning_response = await asyncio.gather(regular_task, reasoning_task)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return regular_response, reasoning_response, duration


# Run synchronous function and print execution time
regular_response, reasoning_response, sync_duration = run_sync()
print(f"""
    Synchronous execution time: {sync_duration:.2f} seconds
    Regular response: {regular_response.content}
    Reasoning response: {reasoning_response.content}
    """)

# Run asynchronous function and print execution time
regular_response, reasoning_response , parallel_duration = asyncio.run(run_parallel())
print(f"""
    Parallel execution time: {parallel_duration:.2f} seconds
    Regular response: {regular_response.content}
    Reasoning response: {reasoning_response.content}
    """)
