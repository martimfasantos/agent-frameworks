import os
import importlib
from typing import Tuple
import argparse

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

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider", 
        type=str, 
        choices=["openai", "azure", "other"], 
        help="The LLM provider to use in the agent."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["response-time", "response-time-loop"], 
        help="Mode. Should be either 'response-time' or 'response-time-loop'"
    )
    parser.add_argument(
        "--iter", 
        type=int, 
        help="Number of iterations. Required if mode is 'response-time-loop'."
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Maintain conversation history in the agent."
    )
    parser.add_argument(
        "--creation",
        action="store_true",
        help="Create a new agent instance each time."
    )
    parser.add_argument(
        "--prints",
        action="store_true",
        help="Print the Assistant's response."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File to save the chat history."
    )
    
    args = parser.parse_args()

    if args.mode is None:
        args.prints = True

    if args.mode == "response-time-loop" and args.iter is None:
        parser.error("--iter is required when --mode is 'response-time-loop'.")
    
    return args

def execute_agent(agent: object, args: argparse.Namespace):
    """
    Execute the agent with the given arguments.
    """
    while True:
        query = input("You: ")

        if query.lower() in ['exit', 'quit']:
            break

        iterations = args.iter if args.mode == "response-time-loop" else 1
        response_times = []

        if args.mode in ["response-time", "response-time-loop"]:
            import time
            import numpy as np

            for _ in range(iterations):
                start = time.time()
                if args.creation:
                    Agent = type(agent) 
                    # print("new agent created")
                    agent = Agent(provider=args.provider, memory=not args.no_memory)
                response = agent.chat(query)
                end = time.time()
                response_times.append(end - start)
                if args.prints:
                    if args.file:
                        with open(args.file, "a") as f:
                            f.write(f"Assistant: {response}\n")
                    else:
                        print(f"Assistant: {response}\n")
            
            print(
                f"{'-'*50}\n"
                f"Mode: {args.mode}\n"
                f"Iterations: {iterations}\n"
                f"\033[92mResponse Time: {np.mean(response_times):.2f} ± {np.std(response_times):.2f}s\033[0m\n"
                f"{'-'*50}"
            )

        else:
            response = agent.chat(query)
            if args.prints:
                print(f"Assistant: {response}")

