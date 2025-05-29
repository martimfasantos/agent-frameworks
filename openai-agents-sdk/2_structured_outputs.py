import os
import asyncio
from pydantic import BaseModel

from agents import Agent, Runner, RunResult, function_tool
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Output models for tools responses
- Output types for agents
- Internal messages

This example shows the usage of output models to structure the responses
from tools, and how to filter sensitive data from the tool output using
output types in the agents. The agent will return a filtered user profile 
without sensitive data like location or IBAN, while still allowing the 
tool to return all data.
-----------------------------------------------------------------------------
"""

# 1. Define the output model for the user profile
class UserProfile(BaseModel):
    id: int
    name: str
    age: int
    location: str
    iban: str

# 2. Define a filtered output model for the user profile
class FilteredUserProfile(BaseModel):
    name: str
    age: int

# 3. Define a function tool to get user information
@function_tool
def get_user_profile(id: int) -> UserProfile:
    print("[debug] get_user_profile called")
    return UserProfile(
        id=id,
        name="Martim Santos",
        age=24,
        location="Lisbon, Portugal",
        iban="DE89370400440532013000"  # Example IBAN
    )

# 4. Define the agent with the function tool registered
agent = Agent(
    name="Example Agent",
    instructions="You are a helpful agent.",
    tools=[get_user_profile],
    output_type=FilteredUserProfile,  # Specify the output model which filters sensitive data
)

async def main():
    
    # 5. Run the agent
    result: RunResult = await Runner.run(
        agent, input="Who is user 1 and what's his location?"
    ) # Output model shouldn't contain sensitive data like location
    
    print("Internal messages:")
    for item in result.new_items:
        print(f"\t{item}")
    print("-" * 50)
    print(f"Final response:\n\t{result.final_output}")
    
    """ 
    Expected final output: 
        name='Martim Santos' age=24
    """


if __name__ == "__main__":
    asyncio.run(main())