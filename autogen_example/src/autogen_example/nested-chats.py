import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_example.settings import settings

# AUTOGEN 0.4 - WORKING!

async def main():
    model_client = OpenAIChatCompletionClient(
        model=settings.openai_model_name, 
        api_key=settings.openai_api_key.get_secret_value()
    )
    	
    # Create the agents for the inner team
    # TODO: the writers are not being called in parallel because 
    # the society of mind agents are not being called in parallel - ref. line 73
    writers = []
    for i in range(8):
        writer = AssistantAgent(
            f"writer{i}", 
            model_client=model_client, 
            system_message="You are a writer, write well."
        )
        writers.append(writer)
    
    editors = []
    for i in range(2):
        editor = AssistantAgent(
            f"editor{i}", 
            model_client=model_client, 
            system_message='''
                You are joke evaluator. Briefly evaluate the jokes you received.
                Then, clearly identify the best joke, print it and write "TERMINATE" to conclude.
            '''
        )
        editors.append(editor)

    inner_termination = TextMentionTermination("TERMINATE")
    
    # Create the inner team(s)
    inner_team1 = RoundRobinGroupChat(
        participants=writers[:4] + [editors[0]], 
        termination_condition=inner_termination
    )
    inner_team2 = RoundRobinGroupChat(
        participants=writers[4:] + [editors[1]], 
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

    translator = AssistantAgent(
        "translator", 
        model_client=model_client, 
        system_message="Translate the text to Spanish."
    )

    # Create the "outer" team
    # TODO: The society of mind agents are not being called in parallel - FIX THIS!
    team = RoundRobinGroupChat(
        participants=[society_of_mind_agent1, society_of_mind_agent2, translator], 
        max_turns=2
    )

    stream = team.run_stream(task="Write a one liner joke with a surprising ending.")
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())