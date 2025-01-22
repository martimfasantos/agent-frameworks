import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.chat_models import ChatOpenAI
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from dotenv import load_dotenv
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

@CrewBase
class ChatBot():
	"""ChatBot crew"""

	load_dotenv()

	llm = ChatOpenAI(
		model=os.getenv("OPENAI_MODEL_NAME"), 
		temperature=0.7, # increase to be more creative
		api_key=os.getenv("OPENAI_API_KEY")
	)

	# defined automatically by the @CrewBase decorator
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def user_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['user_agent'],
			llm=self.llm,
			verbose=True
		)

	@agent
	def database_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['database_agent'],
			# cache=True, # Not sure if this is needed
			# Very buggy, its better to go via Tools
			# knowledge_sources=[
			# 	CrewDoclingSource(
			# 		file_paths=["user1.md", "user2.md"],
			# 	)
			# ],
			tools=[
				DirectoryReadTool(directory='./knowledge'),
		  		FileReadTool() # is able to read all files in the directory above
		  		],
			llm=self.llm,
			verbose=True
		)
	
	@agent
	def data_processing_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['data_processing_agent'],
			llm=self.llm,
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def user_interaction_task(self) -> Task:
		return Task(
			config=self.tasks_config['user_interaction_task'],
		)

	@task
	def database_query_task(self) -> Task:
		return Task(
			config=self.tasks_config['database_query_task'],
		)
	
	@task
	def data_processing_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_processing_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ChatBot crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			memory=False,

			# manager_llm=ChatOpenAI(
			# 	temperature=0, 
			# 	model=os.getenv("OPENAI_MODEL_NAME"), 
			# 	api_key=os.getenv("OPENAI_API_KEY")
			# ),
    		# process=Process.hierarchical, # is a bit buggy and with pydantic errors
			verbose=True,
			# planning=True,
		)
