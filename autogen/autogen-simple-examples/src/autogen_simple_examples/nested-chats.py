import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_simple_examples.settings import settings

# AUTOGEN 0.4 - WORKING!

# -------------------------------------------------------------------------------
# -- GOAL: Evaluate Autogen's ability to parallelize agent calling using the Society of Mind architecture.
# -- DESCRIPTION: Parallelize the nested chats (inner teams) to maximize the number of jokes 
# generated and evaluated in the least amount of time. 
# -------------------------------------------------------------------------------

async def main():
    model_client = OpenAIChatCompletionClient(
        model=settings.openai_model_name, 
        api_key=settings.openai_api_key.get_secret_value()
    )
    	
    # Create the agents for the inner team
    # TODO: the writers are not being called in parallel because 
    # the society of mind agents are not being called in parallel - ref. line 100
    writers = []
    for i in range(8):
        writer = AssistantAgent(
            f"writer{i}", 
            model_client=model_client, 
            system_message="You are a joke writer, write creative and funny jokes."
        )
        writers.append(writer)
    
    evaluators = []
    for i in range(2):
        evaluator = AssistantAgent(
            f"evaluator{i}", 
            model_client=model_client, 
            system_message='''
                You are joke evaluator. Briefly evaluate the jokes you received.
                Then, clearly identify the best joke, print it and write "TERMINATE" to conclude.
            '''
        )
        evaluators.append(evaluator)

    inner_termination = TextMentionTermination("TERMINATE")
    
    # Create the inner team(s)
    inner_team1 = RoundRobinGroupChat(
        participants=writers[:4] + [evaluators[0]], 
        termination_condition=inner_termination
    )
    inner_team2 = RoundRobinGroupChat(
        participants=writers[4:] + [evaluators[1]], 
        termination_condition=inner_termination
    )

    # Create the Society of Mind agent(s)
    society_of_mind_agent1 = SocietyOfMindAgent(
        "society_of_mind_agent1", 
        team=inner_team1, # pass the inner team here
        model_client=model_client,
    )
    society_of_mind_agent2 = SocietyOfMindAgent(
        "society_of_mind_agent2", 
        team=inner_team2, # pass the inner team here
        model_client=model_client,
    )

    translator = AssistantAgent(
        "translator", 
        model_client=model_client, 
        system_message="Translate the text to Spanish."
    )
    
    orchestrator = AssistantAgent(
        "orchestrator",
        description="An agent for planning and coordinating the team's actions.", 
        model_client=model_client, 
        system_message="""
            You are the orchestrator.
            Your job is to break down complex tasks into smaller, manageable subtasks.
            Your team members are:
                writer0-9: Generate one funny joke.
                evaluators0-1: evaluate the generated joke for the writers in their team.
                translator: translates the two best jokes to spanish.

            You only plan and delegate tasks - you do not execute them yourself. You must call all agents in parallel to generate
            the most ammount of jokes in the least ammount of time.

            When assigning tasks, use this format:
            1. <agent> : <task>
            2. <agent> : <task>
            ...

            After all tasks are complete, summarize the findings and end with "TERMINATE".
        """
    )

    # Create the "outer" team
    # TODO: The society of mind agents are not being called in parallel - FIX THIS!
    team = SelectorGroupChat(
        participants=[orchestrator, society_of_mind_agent1, society_of_mind_agent2, translator], 
        model_client=model_client,
        max_turns=10
    )

    stream = team.run_stream(task="Write a one liner joke with a surprising ending.")
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())