import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    LiteLLMModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    prompts,
)
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
# model = HfApiModel(model_id="meta-llama/Llama-3.1-8B-Instruct")
# model = HfApiModel()
model = LiteLLMModel(model_id=os.getenv("OPENAI_MODEL_NAME")) # use if you have a api key (much faster)

# Research Agent
research_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=10,
)

managed_research_agent = ManagedAgent(
    agent=research_agent,
    name="super_researcher",
    description="Researches topics thoroughly using web searches and content scraping. Provide the research topic as input.",
)

# Research Checker Agent
research_checker_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_research_checker_agent = ManagedAgent(
    agent=research_checker_agent,
    name="research_checker",
    description="Checks the research for relevance to the original task request. If the research is not relevant, it will ask for more research.",
)

# Writer Agent
writer_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_writer_agent = ManagedAgent(
    agent=writer_agent,
    name="writer",
    description="Writes blog posts based on the checkedresearch. Provide the research findings and desired tone/style.",
)

# Copy Editor Agent
copy_editor_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_copy_editor = ManagedAgent(
    agent=copy_editor_agent,
    name="editor",
    description="Reviews and polishes the blog post based on the research and original task request. Order the final blog post and any lists in a way that is most engaging to someone working in AI. Provides the final, edited version in markdown.",
)

# Main Blog Writer Manager (like CrewAI, manager agent is the main agent that coordinates the other agents in a "crew")
blog_manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_research_agent, managed_research_checker_agent, managed_writer_agent, managed_copy_editor],
    additional_authorized_imports=["re"],
    # system prompt is added automatically if not provided (default is prompts.CODE_SYSTEM_PROMPT)

    system_prompt=prompts.CODE_SYSTEM_PROMPT + 
    """You are a blog post creation manager. You will be given a task to solve as best you can. Coordinate between research, writing, and editing teams.
    Follow these steps:
    1. Use research_agent to gather information
    2. Pass research to research_checker_agent to check for relevance
    3. Pass research to writer_agent to create the initial draft
    4. Send draft to editor for final polish (guarantee that is engaging to someone working in AI and well formatted)
    4. Return the results as a string in markdown format.
    """
)

def write_blog_post(topic, output_file="outputs/blog_post.md"):
    """
    Creates a blog post on the given topic using multiple agents
    
    Args:
        topic (str): The blog post topic or title
        output_file (str): The filename to save the markdown post
    """
    result = blog_manager.run(f"""Create a blog post about: {topic}
    1. First, research the topic thoroughly, focus on specific products and sources
    2. Then, write an engaging blog post not just a list
    3. Finally, edit and polish the content
    """)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"Blog post has been saved to {output_file}")
    
    return result

# print(blog_manager.system_prompt_template)
topic = "Create a blog post about the top 5 products released at CES 2025. Please include specific product names and sources/reference links."
print(topic)
write_blog_post(topic)
