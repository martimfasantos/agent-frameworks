from __future__ import annotations as _annotations # require for to call the next node in the graph
import asyncio
from dataclasses import dataclass, field
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_ai_examples.settings import settings

# DIAGRAM:
#
#      (Start)
#         ↓
#    +------------+
#    | WriteEmail |
#    +------------+
#      ↓   ↺   ↑ 
#    +------------+
#    |  Feedback  |
#    +------------+
#         ↓
#       (End)
#
# Note: A more detailed diagram depicted in 'diagrams/' directory
#
# Reference: https://ai.pydantic.dev/graph/#genai-example

@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)

# client = AsyncAzureOpenAI(
#     azure_endpoint='...',
#     api_version='2024-07-01-preview',
#     api_key='your-api-key',
# )

# model = OpenAIModel('gpt-4o', openai_client=client)

model = OpenAIModel(
    model_name=settings.openai_model_name,
    api_key=settings.openai_api_key.get_secret_value()
)

email_writer_agent = Agent(
    model=model, # cannot use models.KnownModelName here because of the api key
    result_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(ctx.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(ctx.state.user)}'
            )
        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.all_messages()
        return Feedback(result.data) # type is Email; directly calls the Feedback node


# Outputs from the feedback agent
class EmailRequiresWrite(BaseModel):
    feedback: str

class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    model=model,
    result_type=EmailRequiresWrite | EmailOk,
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.data.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    # define the graph
    feedback_graph = Graph(
        nodes=(WriteEmail, Feedback)
    )
    # run the graph
    final_response, history = await feedback_graph.run(WriteEmail(), state=state) # specify the start node abd the initial state
    for message in history:
        print(message)
        print('-'*50)
    
    # final output
    print('\n\nFinal Email:')
    print(final_response)


if __name__ == '__main__':
    asyncio.run(main())