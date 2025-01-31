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
    # TODO: the writers are not being called in parallel - FIX THIS!
    writers = []
    for i in range(8):
        writer = AssistantAgent(
            f"writer{i}", 
            model_client=model_client, 
            system_message="You are a writer, write well."
        )
        writers.append(writer)
        
    editor = AssistantAgent(
        "editor",
        model_client=model_client,
        system_message='''
            You are joke evaluator. Briefly evaluate the jokes you received.
            Then, clearly identify the best joke, print it and write "TERMINATE" to conclude.
        '''
    )

    inner_termination = TextMentionTermination("TERMINATE")
    # Create the inner team
    inner_team = RoundRobinGroupChat(
        participants=writers + [editor], 
        termination_condition=inner_termination
    )

    society_of_mind_agent = SocietyOfMindAgent(
        "orchestrator", 
        team=inner_team, # pass the inner team here
        model_client=model_client,
    )

    agent3 = AssistantAgent(
        "translator", 
        model_client=model_client, 
        system_message="Translate the text to Spanish."
    )

    # Create the "outer" team
    team = RoundRobinGroupChat([society_of_mind_agent, agent3], max_turns=2)

    stream = team.run_stream(task="Write a one liner joke with a surprising ending.")
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())