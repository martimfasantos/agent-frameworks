import asyncio
from dataclasses import dataclass
from email import message
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai_examples.settings import settings


model = OpenAIModel(
    model_name=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value()
)


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123 and include_pending:
            return 123.45
        else:
            raise ValueError('Customer not found')


@dataclass
class SupportDependencies:  
    customer_id: int
    db: DatabaseConn  


class SupportResult(BaseModel):  
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  
    model=model, # cannot use models.KnownModelName here because of the api key 
    deps_type=SupportDependencies,
    result_type=SupportResult,  
    system_prompt=(  
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)

# system prompt runs during runtime when the agent is called
@support_agent.system_prompt  
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool(retries=1)
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn()) # context to the agent
    result = await support_agent.run('What is my balance?', deps=deps)  
    print(result.data) # note that the data is a SupportResult object
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """
    
    print("-"*50)
    for message in result.all_messages():
        print(f"{message}\n")
    """
    1. Model Request (SystemPromptPart(static), SystemPromptPart(injected), UserPromptPart)
    2. Model Response (ToolCallPart - customer_balance - args="include_pending":true (user_id is passed via ctx))
    3. Model Request (ToolReturnPart - customer_balance - content=123.45)
    4. Model Response (ToolReturnPart - final_result - args='{"support_advice":"Your current account balance is $123.45.","block_card":false,"risk":0}')
    5. Model Request (ToolReturnPart - final_result - content='Final result processed.')
    """
    print(f"Total usage: {result._usage}")
    print("-"*50)


    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """


if __name__ == '__main__':
    asyncio.run(main())