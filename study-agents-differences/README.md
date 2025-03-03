# study-agents-differences

First, create a python virtual environment with:

```bash
python3 -m venv .venv
```

Then, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install the dependencies with:

```bash
pip install -r requirements.txt 
```

Create a `.env` file in the root of the project based on the `.env.example` file.


Now, you can run the examples by running each file.

## Agents

Each agent can be run by executing the corresponding file.
Some flags can be passed to the agents to change their behavior when running independently.

### Flags
- `--mode [mode]`: The mode in which the agent will run. It can be None, `response-time` or `response-time-loop`. Default is None, which will run a simple chatbot interface in the terminal.
- `--iter [int]`: Number of iterations the agent will run. Default is 1. (Only needed for `response-time-loop` mode).
- `--creation`: If you want to create an Agent's instance in every iteration. Default is False.
- `--no-memory`: If you want to run the agent without memory. Default is False.
- `--print`: If you want to see the agent's responses. Default is True for Normal mode and False for `response-time` and `response-time-loop` modes.
- `--file [output file]`: If you want to save the agent's responses to a file. Default is False. (Only needed for `response-time` and `response-time-loop` modes)

If you want to see the `web_search_tool` call and response, uncomment the print statements in the `web_search_tool` function in the respective agent file.

## UI

To run the streamlit UI, run the following command:

```bash
streamlit run agent-ui.py
```

This will open a new tab in your browser with the UI where you can interact with the agents.