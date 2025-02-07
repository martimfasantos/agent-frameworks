import os
from crewai import Agent, Task, Crew, Process
from agents import EmailWriter, DTCCMOAgent, CopywriterAgent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

# llm = ChatGroq(model_name="groq/llama3-8b-8192", # change this to your preferred model (https://console.groq.com/settings/limits)
#                temperature=0.5,
#                groq_api_key=os.getenv("GROQ_API_KEY")
#                )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

email_writer = EmailWriter(llm)
dtc_cmo = DTCCMOAgent(llm) # this agent has access to human tools <--
copywriter = CopywriterAgent(llm)

# Define Task
email_task = Task(
    description='''
    1. Write three variations of a cold email selling a video editing solution. 
    Ask Human for advice on how to write a cold email.
    2. Critique the written emails for effectiveness and engagement.
    3. Proofread the emails for grammatical correctness and clarity.
    4. Adjust the emails to ensure they meet cold outreach best practices. make sure to take into account the feedback from human 
    which is a tool provided to dtc_cmo.
    5. Rewrite the emails based on all feedback to create three final versions.''',
    agent=dtc_cmo  # DTC CMO is in charge and can delegate
)

# Create a Single Crew
email_crew = Crew(
    agents=[email_writer, dtc_cmo, copywriter],
    tasks=[email_task],
    verbose=True,
    process=Process.sequential
)

# Execution Flow
print("Crew: Working on Email Task")
final_emails = email_crew.kickoff()