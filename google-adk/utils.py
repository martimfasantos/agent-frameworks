from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME="my_agent"
USER_ID="user"
SESSION_ID="1234"

# Session and Runner
async def setup_session_and_runner(agent: Agent):
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(
    agent: Agent, 
    query: str,
    *,
    tool_calls: bool = True,
    tool_call_results: bool = False,
):
    content = types.Content(
        role='user', 
        parts=[types.Part(text=query)]
    )
    _, runner = await setup_session_and_runner(agent)
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    async for event in events:
        if tool_calls:
            handle_tool_calls(event)
        if tool_call_results:
            handle_tool_responses(event)
        
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

def handle_tool_calls(event):
    calls = event.get_function_calls()
    if calls:
        for call in calls:
            tool_name = call.name
            arguments = call.args  # This is usually a dictionary
            print(f"  Tool: {tool_name}, Args: {arguments}")
    else:
        print("No tool calls found.")

def handle_tool_responses(event):
    responses = event.get_function_responses()
    if responses:
        for response in responses:
            tool_name = response.name
            result_dict = response.response  # The dictionary returned by the tool
            print(f"  Tool Result: {tool_name} -> {result_dict}")
    else:
        print("No tool responses found.")