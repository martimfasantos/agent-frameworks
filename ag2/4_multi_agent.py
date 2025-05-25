import os

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
----------------------------------------------------------------------
In this example, we explore AG2's Agents with the following features:
- Multi-agent collaboration
- Multi-agent orchestration
- Custom agent system prompts and descriptions
- Agent termination conditions

This example shows 
----------------------------------------------------------------------
"""

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

# 1. Define the system prompts and descriptions for 
# the planner and reviewer agents
# 1.1 System prompts
planner_system_prompt = """
You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

reviewer_system_prompt = """
You are a classroom lesson reviewer.
You compare the lesson plan to the fourth grade curriculum and provide a maximum of 3 recommended changes.
Provide only one round of reviews to a lesson plan.
"""

# 1.2 Descriptions
planner_description = "Creates or revises lesson plans."

reviewer_description = """Provides one round of reviews to a lesson plan
for the lesson_planner to revise."""

# 2. Create the lesson planner and reviewer agents with 
# their system messages and descriptions
with llm_config:
    lesson_planner = ConversableAgent(
        name="planner_agent",
        system_message=planner_system_prompt,
        description=planner_description,
    )

    lesson_reviewer = ConversableAgent(
        name="reviewer_agent",
        system_message=reviewer_system_prompt,
        description=reviewer_description,
    )

# 3. Create the teacher agent with a system prompt
teacher_system_prompt = """
You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
When you are happy with a lesson plan, output "DONE!".
"""

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_system_prompt,
    # Our teacher can end the conversation by saying DONE!
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    llm_config=llm_config,
)

# 4. Create the GroupChat with agents and selection method
groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto",
    messages=[],
)

# 5. Our GroupChatManager will manage the conversation and 
# uses an LLM to select the next agent
manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config,
)

# 6. Initiate the chat with the GroupChatManager as the recipient
teacher.initiate_chat(
    recipient=manager,
    message="Today, let's introduce our kids to the solar system."
)
# Expected workflow:
#   Teacher -> Planner --> Planner creates a lesson plan --> 
#   --> Reviewer --> Reviewer reviews the lesson plan --> Planner -->
#   --> Incorporates the review suggestions --> Teacher --> DONE!
