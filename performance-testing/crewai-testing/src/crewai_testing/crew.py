import os
from logging import getLogger
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, task, crew
from langchain_openai import ChatOpenAI
from .tools.geometric_mean_tool import GeometricMeanTool
from .settings import settings
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
)

agent_logger = getLogger("agent")

@CrewBase
class ChatBot():
	"""ChatBot crew"""

	llm = ChatOpenAI(
		model=settings.openai_model_name, 
		api_key=settings.openai_api_key.get_secret_value(),
		temperature=settings.temperature,
		max_tokens=settings.max_tokens,
	)

	# defined automatically by the @CrewBase decorator
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	
	@agent
	def user_agent(self) -> Agent:
		agent_logger.info("Creating User Agent")
		return Agent(
			config=self.agents_config['user_agent'],
			llm=self.llm,
			verbose=True
		)

	@agent
	def database_agent(self) -> Agent:
		agent_logger.info("Creating Database Agent with Tools")
		return Agent(
			config=self.agents_config['database_agent'],
			tools=[
				DirectoryReadTool(directory='./knowledge'),
		  		FileReadTool(), # is able to read all files in the directory above
		  	],
			llm=self.llm,
			verbose=True
		)
	
	@agent
	def data_processing_agent(self) -> Agent:
		agent_logger.info("Creating Data Processing Agent")
		return Agent(
			config=self.agents_config['data_processing_agent'],
			llm=self.llm,
			verbose=True
		)
	
	@agent
	def geometric_mean_agent(self) -> Agent:
		agent_logger.info("Creating Geometric Mean Agent")
		return Agent(
			config=self.agents_config['geometric_mean_agent'],
			tools=[
				GeometricMeanTool(),
			],
			llm=self.llm,
			verbose=True
		)

	
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
	
	@task
	def geometric_mean_task(self) -> Task:
		return Task(
			config=self.tasks_config['geometric_mean_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ChatBot crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			# verbose=True,
		)
