import os
import importlib
import tiktoken
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
        choices=["azure", "openai", "other"],
        default="azure", 
        help="The LLM provider to use in the agent."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["metrics", "metrics-loop"], 
        help="Mode. Should be either 'metrics' or 'metrics-loop'"
    )
    parser.add_argument(
        "--iter", 
        type=int, 
        help="Number of iterations. Required if mode is 'metrics-loop'."
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Maintain conversation history in the agent."
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new agent instance each time."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the Agent's logs and messages."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File to save the chat history."
    )
    
    args = parser.parse_args()

    if args.mode is None:
        args.verbose = True

    if args.mode == "metrics-loop" and args.iter is None:
        parser.error("--iter is required when --mode is 'metrics-loop'.")
    
    return args

def execute_agent(agent: object, args: argparse.Namespace):
    """
    Execute the agent with the given arguments.
    """
    while True:
        query = input("You: ")

        if query.lower() in ['exit', 'quit']:
            break

        iterations = args.iter if args.mode == "metrics-loop" else 1
        response_times = []
        tokens_counter = {
            "total_embedding_token_count": 0,
            "prompt_llm_token_count": 0,
            "completion_llm_token_count": 0,
            "total_llm_token_count": 0
        }

        if args.mode in ["metrics", "metrics-loop"]:
            import time
            import numpy as np

            for _ in range(iterations):
                if args.create:
                    Agent = type(agent) 
                    agent = Agent(
                        provider=args.provider, 
                        memory=False if args.no_memory else True,
                        verbose=args.verbose,
                        tokens=True
                    )
                    if args.verbose:
                        print("New agent created.")
                response, exec_time, token_counter = agent.chat(query)
                # Add time taken to respond and update token counts
                response_times.append(exec_time)
                for key, value in token_counter.items():
                    tokens_counter[key] += value
                
                if args.verbose:
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
                f"{'-'*50}\n"
                f"Embedding Tokens: {(tokens_counter["total_embedding_token_count"]/iterations):.1f}\n"
                f"LLM Prompt Tokens: {(tokens_counter["prompt_llm_token_count"]/iterations):.1f}\n"
                f"LLM Completion Tokens: {(tokens_counter["completion_llm_token_count"]/iterations):.1f}\n"
                f"\033[36mTotal LLM Token Count: {(tokens_counter["total_llm_token_count"]/iterations):.1f}\033[0m\n"
                f"{'-'*50}\n"
            )

            # Reset times and token counts
            response_times = []
            tokens_counter = {
                "total_embedding_token_count": 0,
                "prompt_llm_token_count": 0,
                "completion_llm_token_count": 0,
                "total_llm_token_count": 0
            }

        else:
            response = agent.chat(query)
            if args.verbose:
                print(f"Assistant: {response}")



def get_tokens(
    output_data, input_messages=None, tools=None, encoding_name: str = "cl100k_base"
) -> tuple[int, int]:
    if "usage" in output_data:
        input_tokens = output_data["usage"].get("prompt_tokens", 0)
        output_tokens = output_data["usage"].get("completion_tokens", 0)
        return input_tokens, output_tokens
    encoding = tiktoken.get_encoding(encoding_name)
    input_tokens = 0
    output_tokens = 0
    # Count input message tokens
    if input_messages:
        for message in input_messages:
            print(message)
            for key, value in message.items():
                input_tokens += len(encoding.encode(str(value)))
    # Count tool definitions as input tokens
    if tools:
        input_tokens += len(encoding.encode(str(tools)))
    # Count output message tokens
    if "choices" in output_data:
        for choice in output_data["choices"]:
            if "message" in choice:
                message = choice["message"]
                # Count normal message content tokens
                if "content" in message and message["content"] is not None:
                    output_tokens += len(encoding.encode(message["content"]))
                # Count tool call tokens
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        output_tokens += len(encoding.encode(str(tool_call)))
    return input_tokens, output_tokens