import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat, Swarm
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_example.settings import settings

# AUTOGEN 0.4 - WORKING!

# Issue: https://github.com/microsoft/autogen/issues/5359

def calculator(a: float, b: float, operator: str) -> str:
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == '/':
            if b == 0:
                return 'Error: Division by zero'
            return str(a / b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'
    
def calculator2(a: float, b: float, operator: str) -> str:
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == '/':
            if b == 0:
                return 'Error: Division by zero'
            return str(a / b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'

async def main():
    model_client = OpenAIChatCompletionClient(
        model=settings.openai_model_name, 
        api_key=settings.openai_api_key.get_secret_value(),
        parallel_tool_calls=True # This is the key to parallelization
    )
    
    	
    assistant1 = AssistantAgent(
        "ParallelizationAssistant1", 
        model_client=model_client, 
        tools=[calculator],
        handoffs=["planning_agent"],
        system_message="You solve the problem in parallel with another assistant."
    )
    assistant2 = AssistantAgent(
        "ParallelizationAssistant2", 
        model_client=model_client, 
        tools=[calculator2],
        handoffs=["planning_agent"],
        system_message="You solve the problem in parallel with another assistant."
    )

    inner_termination = TextMentionTermination("TERMINATE")
    
    # Create the inner team(s)
    inner_team1 = RoundRobinGroupChat(
        participants=[assistant1], 
        termination_condition=inner_termination
    )
    inner_team2 = RoundRobinGroupChat(
        participants=[assistant2], 
        termination_condition=inner_termination
    )

    # Create the Society of Mind agent(s)
    society_of_mind_agent1 = SocietyOfMindAgent(
        "orchestrator1", 
        team=inner_team1, # pass the inner team here
        model_client=model_client,
    )
    society_of_mind_agent2 = SocietyOfMindAgent(
        "orchestrator2", 
        team=inner_team2, # pass the inner team here
        model_client=model_client,
    )

    planning_agent = AssistantAgent(
        "planning_agent", 
        model_client=model_client, 
        handoffs=["orchestrator1", "orchestrator2"],
        system_message="""
            You are a planning agent.
            You delegate the task to the Society of Mind agents.
            You must call the Society of Mind agents in parallel to solve the problem.
            When the task can be subdivided, you must delegate each sub-task to different agents.
            When delegating the task to the Society of Mind agents, ensure that they are working in parallel.
            The delegation should be done in this format:
            - <agent1>, <task1>
            - <agent2>, <task2>

            When the Society of Mind agents have finished their tasks, you must collect the results and present them to the user.
            When the task is completed, write "TERMINATE" to conclude.
        """
    )

    termination = MaxMessageTermination(
        max_messages=10) | TextMentionTermination("TERMINATE")

    team = Swarm(
        participants=[planning_agent, society_of_mind_agent1, society_of_mind_agent2], 
        termination_condition=termination,
        max_turns=10
    )

    # Results:
    # Planning Agent is recognizing that he has to call both Society of Mind agents in parallel
    # HOWEVER "UserWarning: Multiple handoffs detected only the first is executed: ['transfer_to_orchestrator1', 'transfer_to_orchestrator2']"
    # Is it possible to parellelize?

    stream = team.run_stream(task="What are the results of [545.34567 * 34555.34] and [2 / 2] ")
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())