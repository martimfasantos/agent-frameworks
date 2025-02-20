import random

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai_examples.settings import settings


# Reference: https://ai.pydantic.dev/tools/

model = OpenAIModel(
    model_name=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value()
)

agent = Agent(
    model=model,  
    deps_type=str, # because of the player's name in the ctx.deps
    result_type=str,
    system_prompt=(
        "Call the 'roll_dice' tool twice in parallel."
        "You're a dice game, you should roll the die twice in parallel and see"
        "if any of the numbers you get back matches the user's guess." 
        "If so, tell them they're a winner and both numbers."
        "Use the player's name in the response."
    ),
)


@agent.tool_plain  
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool  
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Anne')
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!

print("-"*50)
for message in dice_result.all_messages():
    print(f"{message}\n")
# ...
# ModelResponse(parts=[
#   ToolCallPart(tool_name='roll_dice', args='{}', tool_call_id='call_S11ZXmXqg1J08DBEfDlGKISD', part_kind='tool-call'), 
#   ToolCallPart(tool_name='roll_dice', args='{}', tool_call_id='call_cQgz7wjMl8etWPBiyqqzOBBG', part_kind='tool-call')], 
#   ...
# )
# ...
#> Model sucessfully called the tool 'roll_dice' twice in parallel.