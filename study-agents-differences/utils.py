import os
import importlib
from typing import Tuple

# Function to get available agent modules and their names
def get_available_agents():
    agents = {}
    for file in os.listdir('.'):
        if file.endswith('_agent.py'):
            module_name = file[:-3]  # Remove .py
            try:
                module = importlib.import_module(module_name)
                temp_agent = module.Agent()
                agents[module_name] = temp_agent.name
            except Exception as e:
                print(f"Error loading {module_name}: {str(e)}")
    return agents

# Generate a list of the available tools
def get_tools_descriptions(tools_tuple: list[Tuple[str, str]]) -> str:
    """
    Generate a list of the available tools.

    Args:
        tools_tuple (list[Tuple[str, str], ...]): A list of (tool name, tool description) pairs.
    """
    return f"{'\n'.join([f'- {tool} ({desc})' for tool, desc in tools_tuple])}"