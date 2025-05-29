import os
import asyncio
import json
from pydantic import BaseModel, Field

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------------------------
In this example, we explore OpenAI's Agent class with the following features:
- Output guardrails

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the 
agent's response contains a phone number.
-------------------------------------------------------------------------
"""

# 1. Define the output model for the agent's response
class MessageOutput(BaseModel):
    reasoning: str = Field(
        description="Thoughts on how to respond to the user's message"
    )
    response: str = Field(
        description="The response to the user's message"
    )
    user_name: str | None = Field(
        description="The name of the user who sent the message, if known"
    )

# 2. Define the output guardrail function
@output_guardrail
async def sensitive_data_check(
    context: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    phone_number_in_response = "650" in output.response
    phone_number_in_reasoning = "650" in output.reasoning

    return GuardrailFunctionOutput(
        output_info={
            "phone_number_in_response": phone_number_in_response,
            "phone_number_in_reasoning": phone_number_in_reasoning,
        },
        tripwire_triggered=phone_number_in_response or phone_number_in_reasoning,
    )

# 3. Define the agent with the output model and output guardrail
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[sensitive_data_check],
)


async def main():

    # 4. Run the agent with a user message without sensitive data
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed - guardrail didn't trip as expected.")

    try:
        # 5. Run the agent with a user message that contains sensitive data
        result = await Runner.run(
            agent, "My phone number is 650-123-4567. Where do you think I live?"
        )
        print(
            f"Guardrail didn't trip - this is unexpected. Output: {
                json.dumps(result.final_output.model_dump(), indent=2)
            }"
        )

    except OutputGuardrailTripwireTriggered as e:
        print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")


if __name__ == "__main__":
    asyncio.run(main())